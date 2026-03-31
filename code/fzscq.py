"""
fzsc_v2.py (Variational Fix Version)
✅ 修复: 解决"回归均值"导致的"怪胎分子"问题
✅ 原理: 升级为变分投影 (Variational Projection)，预测分布而非单点
✅ 兼容: 推理模式下接口不变，无缝适配 gen_seeds.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
import pickle
import numpy as np
import os
import math
from tqdm import tqdm
import selfies as sf
from rdkit import Chem, RDLogger

# 屏蔽 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# ================= 配置 =================
TRANSFORMER_PATH = "results/best_transformer_v4.pth"
GRAPHIT_PATH = "results/graphit_vae_best.pth"
# 请确认数据路径正确
DATA_PATH = r'D:\pythonstu\pythonProject16\dev\new_gat-vae\final_results\final_active_sites_with_graphs_full_data.pkl'
OUTPUT_PATH = "results/fzscq.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4  # 稍微降低学习率，更稳
KL_WEIGHT = 0.001  # KL 散度权重，防止后验塌陷


# ================= 1. 定义网络结构 =================

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x): return x + self.net(x)


class RobustProjectionNetwork(nn.Module):
    """
    升级版投影网络：变分结构 (Variational)
    Train: 返回 (采样z, 均值mu, 方差logvar)
    Eval:  返回 均值mu (确定性输出，适配 gen_seeds.py)
    """

    def __init__(self, input_dim=256, output_dim=256, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )

        # VAE 核心：预测均值和方差
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_var = nn.Linear(hidden_dim, output_dim)

        # Transformer Latent 是 Tanh 激活的，范围 [-1, 1]
        self.act = nn.Tanh()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)

        # 约束均值在 [-1, 1] 之间，匹配 Transformer 的 latent 空间
        mu = self.act(mu)

        if self.training:
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            # 推理模式：直接返回均值，保证 gen_seeds.py 调用时不报错
            return mu


# ================= 2. 辅助类 (用于加载预训练模型) =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300, dropout=0.05):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x): return x + self.pe[:, :x.size(1)].to(x.device)


class TransformerSELFIESAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.05,
                 latent_dim=256, max_length=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_length, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True,
                                                   norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.attn_pool = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1),
                                       nn.Softmax(dim=1))
        self.to_latent = nn.Sequential(nn.Linear(d_model, latent_dim), nn.Tanh())
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, d_model), nn.GELU())
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True,
                                                   norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask=None):
        src_emb = self.embedding(src) * math.sqrt(512)
        src_emb = self.pos_encoder(src_emb)
        if src_mask is None: src_mask = (src == 0)
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_mask)
        weights = self.attn_pool(memory)
        if src_mask is not None:
            mask_expanded = (~src_mask).unsqueeze(-1).float()
            weights = weights * mask_expanded
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        pooled = (memory * weights).sum(dim=1)
        latent = self.to_latent(pooled)
        return latent


class SELFIESTokenizer:
    def __init__(self, vocab_list):
        self.vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        for i, t in enumerate(vocab_list):
            if t not in self.vocab: self.vocab[t] = i + 4
        self.idx2token = {v: k for k, v in self.vocab.items()}

    def encode(self, selfies_str, max_len=300):
        try:
            tokens = list(sf.split_selfies(selfies_str))
        except:
            return [1, 2] + [0] * (max_len - 2)
        ids = [1] + [self.vocab.get(t, 3) for t in tokens] + [2]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids


def create_node_features(aa_types):
    aa_map = {'GLY': 0, 'ALA': 1, 'VAL': 2, 'ILE': 3, 'LEU': 4, 'PHE': 5, 'TRP': 6, 'TYR': 7, 'SER': 8, 'THR': 9,
              'CYS': 10, 'MET': 11, 'ASN': 12, 'GLN': 13, 'ASP': 14, 'GLU': 15, 'LYS': 16, 'ARG': 17, 'HIS': 18,
              'PRO': 19}
    feats = []
    for aa in aa_types:
        f = [0] * 9
        idx = aa_map.get(aa.upper()[:3], -1)
        if idx != -1: f[idx % 9] = 1
        feats.append(f)
    return np.array(feats, dtype=np.float32)


def adj_to_edge_index(adj):
    if adj.size == 0: return np.array([[], []])
    return np.stack(np.where(adj > 0), axis=0)


# ================= 3. 数据与训练 =================
# 引入 os 库检查文件是否存在
import os


def prepare_data(pkl_path, graph_model, transformer, tokenizer, device):
    print(f"📖 读取数据: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    # 打印 Keys 确认一下（调试用）
    if len(raw) > 0:
        print(f"🔍 [Debug] Keys detected: {list(raw[0].keys())}")

    prot_vecs, mol_vecs = [], []
    graph_model.eval();
    transformer.eval()

    print("⏳ 预计算特征中 (Auto-extracting SMILES)...")

    success_count = 0
    fail_reasons = {}

    with torch.no_grad():
        for i, item in enumerate(tqdm(raw)):
            try:
                # ================= 核心修复部分开始 =================
                # 1. 尝试直接获取 SMILES
                smi = item.get('smiles')

                # 2. 如果没有 SMILES，尝试从 'ligand_path' 读取文件转换
                if not smi and 'ligand_path' in item:
                    try:
                        path = str(item['ligand_path'])
                        # 兼容相对路径：如果 path 是相对的，可能需要拼上前缀
                        # 这里假设 path 是绝对路径或者相对于当前运行目录
                        if os.path.exists(path):
                            mol = None
                            if path.endswith('.mol2'):
                                mol = Chem.MolFromMol2File(path)
                            elif path.endswith('.sdf'):
                                mol = Chem.MolFromMolFile(path)
                            elif path.endswith('.pdb'):
                                mol = Chem.MolFromPDBFile(path)

                            if mol:
                                smi = Chem.MolToSmiles(mol)
                        else:
                            # 记录一下文件找不到，方便排查
                            # raise ValueError(f"Ligand file not found: {path}")
                            pass
                    except Exception:
                        pass

                # 3. 如果还是没有 SMILES，那这条数据真没救了
                if not smi:
                    raise ValueError("No SMILES found & Ligand file failed")

                # 4. 尝试 SELFIES 编码
                try:
                    se = sf.encoder(smi)
                    if not se: raise ValueError("SELFIES is None")
                except:
                    raise ValueError("SELFIES Encode Fail")
                # ================= 核心修复部分结束 =================

                # 5. 处理 Protein (保持不变)
                if 'x' in item:
                    x, ei = item['x'], item['edge_index']
                elif 'aa_types' in item:
                    x = torch.FloatTensor(create_node_features(item['aa_types']))
                    ei = torch.LongTensor(adj_to_edge_index(np.array(item['adjacency_matrix'])))
                else:
                    raise ValueError("No Protein Feature")

                x, ei = x.to(device), ei.to(device)
                _, p_mean, _, _ = graph_model(x, ei, x.size(0))

                # 6. 处理 Molecule Latent
                ids = torch.tensor([tokenizer.encode(se)], dtype=torch.long).to(device)
                m_latent = transformer.encode(ids, (ids == 0))

                prot_vecs.append(p_mean.squeeze(0).cpu())
                mol_vecs.append(m_latent.squeeze(0).cpu())
                success_count += 1

            except Exception as e:
                msg = str(e)
                if msg not in fail_reasons: fail_reasons[msg] = 0
                fail_reasons[msg] += 1
                continue

    print(f"✅ 有效数据对: {success_count} / {len(raw)}")

    if success_count == 0:
        print("\n⚠️ 失败原因详情:")
        for r, c in fail_reasons.items():
            print(f"  - {r}: {c} 条")
        print(
            "\n💡 提示: 如果 'No SMILES found' 很多，说明 'ligand_path' 指向的文件路径可能不对（比如是在另一台电脑生成的路径）。")
        raise RuntimeError("无有效数据")

    return TensorDataset(torch.stack(prot_vecs), torch.stack(mol_vecs))


def main():
    # 1. 加载依赖模型
    try:
        from graphit import GraphiTVAE
    except:
        print("❌ 缺少 graphit.py"); return

    logger = logging.getLogger("Alignment")

    # Load Models (Frozen)
    t_ckpt = torch.load(TRANSFORMER_PATH, map_location='cpu')
    tokenizer = SELFIESTokenizer(t_ckpt['vocab_list'])
    transformer = TransformerSELFIESAutoencoder(**t_ckpt['config']).to(DEVICE)
    transformer.load_state_dict(t_ckpt['model_state_dict'], strict=False)

    g_ckpt = torch.load(GRAPHIT_PATH, map_location='cpu')
    graph_model = GraphiTVAE(9, 128, 256, 6).to(DEVICE)
    graph_model.load_state_dict(g_ckpt.get('model_state_dict', g_ckpt))

    # Data
    dataset = prepare_data(DATA_PATH, graph_model, transformer, tokenizer, DEVICE)
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Train
    projection = RobustProjectionNetwork().to(DEVICE)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=LR, weight_decay=1e-4)
    mse_crit = nn.MSELoss()

    best_loss = float('inf')

    print("🚀 开始变分对齐训练...")
    for epoch in range(EPOCHS):
        projection.train()
        train_loss = 0
        for p, m in train_loader:
            p, m = p.to(DEVICE), m.to(DEVICE)

            optimizer.zero_grad()
            z, mu, logvar = projection(p)  # Training 返回 3个值

            # Loss = MSE + KL
            recon_loss = mse_crit(z, m)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / p.size(0)  # Batch mean

            loss = recon_loss + KL_WEIGHT * kl_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        projection.eval()
        val_loss = 0
        with torch.no_grad():
            for p, m in val_loader:
                p, m = p.to(DEVICE), m.to(DEVICE)
                mu = projection(p)  # Eval 只返回 mu
                val_loss += mse_crit(mu, m).item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d}: Train Loss {train_loss / len(train_loader):.4f}, Val MSE {val_loss / len(val_loader):.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            # 保存整个 state_dict
            torch.save({'projection': projection.state_dict()}, OUTPUT_PATH)

    print(f"✅ 训练完成! 模型已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()