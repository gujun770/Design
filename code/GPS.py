import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, degree, softmax
from torch_geometric.nn import MessagePassing
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import time
import json
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 唯一的改动：启用CUDA优化（可选，不影响稳定性）==========
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


# ====================== GraphiT架构======================

class GraphInteractionModule(MessagePassing):
    """图交互模块 - GraphiT的核心创新"""

    def __init__(self, hidden_dim, num_heads=8, edge_dim=None):
        super().__init__(aggr='add', node_dim=0)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 查询、键、值投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 边特征投影（如果有）
        if edge_dim:
            self.edge_proj = nn.Linear(edge_dim, num_heads)
        else:
            self.edge_proj = None

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 门控机制
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_attr=None):
        """
        x: [N, hidden_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_dim] (optional)
        """
        # 多头投影
        Q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        # 边特征处理
        if self.edge_proj and edge_attr is not None:
            edge_bias = self.edge_proj(edge_attr)  # [E, num_heads]
        else:
            edge_bias = None

        # 消息传递
        out = self.propagate(edge_index, Q=Q, K=K, V=V, edge_bias=edge_bias, size=None)
        out = out.view(-1, self.hidden_dim)

        # 门控机制
        gate_input = torch.cat([x, out], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))

        # 残差连接与门控
        out = gate * out + (1 - gate) * x

        return self.dropout(out)

    def message(self, Q_i, K_j, V_j, edge_bias, index, ptr, size_i):
        # 计算注意力分数
        attn = (Q_i * K_j).sum(dim=-1) / math.sqrt(self.head_dim)

        # 添加边偏置
        if edge_bias is not None:
            attn = attn + edge_bias

        # Softmax
        attn = softmax(attn, index, ptr, size_i)
        attn = self.dropout(attn)

        # 应用注意力权重
        msg = V_j * attn.unsqueeze(-1)

        return msg


class GraphiTLayer(nn.Module):
    """GraphiT层：结合局部和全局信息"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()

        # 局部交互（消息传递）
        self.local_interaction = GraphInteractionModule(hidden_dim, num_heads)

        # 全局注意力
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch_mask=None):
        """
        x: [N, hidden_dim] or [B, N, hidden_dim]
        edge_index: [2, E]
        """
        # 处理批次维度
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, hidden_dim]
            single_graph = True
        else:
            single_graph = False

        B, N, D = x.shape

        # 局部交互（在图结构上）
        x_flat = x.view(-1, D)  # [B*N, D]
        local_out = self.local_interaction(x_flat, edge_index)
        local_out = local_out.view(B, N, D)

        # 全局注意力
        x_norm = self.norm1(x)
        global_out, _ = self.global_attention(x_norm, x_norm, x_norm)

        # 融合局部和全局信息
        combined = torch.cat([local_out, global_out], dim=-1)
        fused = self.fusion(combined)
        x = x + self.dropout(fused)

        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))

        if single_graph:
            x = x.squeeze(0)  # [N, hidden_dim]

        return x


class GraphiTEncoder(nn.Module):
    """GraphiT编码器"""

    def __init__(self, node_features=9, hidden_dim=128, latent_dim=256,
                 num_layers=6, num_heads=8, dropout=0.1, max_nodes=50):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(node_features, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 可学习的位置编码
        self.pos_encoder = nn.Embedding(max_nodes, hidden_dim)

        # 度编码
        self.degree_encoder = nn.Embedding(max_nodes, hidden_dim)

        # 中心性编码
        self.centrality_encoder = nn.Linear(1, hidden_dim)

        # GraphiT层
        self.layers = nn.ModuleList([
            GraphiTLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 读出层（多尺度池化）
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # VAE头
        self.mean_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.logvar_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

    def compute_centrality(self, edge_index, num_nodes):
        """计算节点中心性特征"""
        if edge_index.numel() == 0:
            return torch.zeros(num_nodes, 1, device=edge_index.device)

        # 度中心性
        row, col = edge_index
        deg = degree(row, num_nodes)

        # 归一化
        deg = deg / (num_nodes - 1 + 1e-8)

        return deg.unsqueeze(-1)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        device = x.device

        # 输入投影
        h = self.input_proj(x)

        # 位置编码 (纯粹的节点级标识)
        pos_ids = torch.arange(num_nodes, device=device)
        h_init = h + self.pos_encoder(pos_ids)  # <--- 【修改点：保存干净的节点身份特征】

        h = h_init  # 继续后续的图结构运算


        if edge_index.numel() > 0:
            row, col = edge_index
            deg = degree(row, num_nodes, dtype=torch.long)
            deg = torch.clamp(deg, max=self.max_nodes - 1)
        else:
            deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
        h = h + self.degree_encoder(deg)

        # 中心性编码
        centrality = self.compute_centrality(edge_index, num_nodes)
        h = h + self.centrality_encoder(centrality)

        # 应用GraphiT层 (消息传递，最严重的泄露源)
        for layer in self.layers:
            h = layer(h, edge_index)

        # 多尺度读出 (计算全局特征 h_global 的逻辑不变)
        if h.dim() == 2:
            h_mean = h.mean(dim=0, keepdim=True)
            h_max = h.max(dim=0, keepdim=True)[0]
            weights = F.softmax(centrality.squeeze(), dim=0)
            h_weighted = (h * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
            h_global = torch.cat([h_mean, h_max, h_weighted], dim=-1)
            h_global = self.readout_mlp(h_global)
        else:
            h_global = h.mean(dim=1)

        mean = self.mean_proj(h_global)
        logvar = self.logvar_proj(h_global)
        logvar = torch.clamp(logvar, min=-10, max=2)


        return h_init, mean, logvar


# ========== 唯一重要的改动：优化解码器（去除双重for循环）==========

class ImprovedDecoder(nn.Module):
    """真正向量化的解码器：保持原版逻辑，但用矩阵运算"""

    def __init__(self, latent_dim=256, hidden_dim=128):
        super().__init__()

        # 潜在空间到节点特征（完全不变）
        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 节点特征细化（完全不变）
        self.node_refine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 边预测MLP（完全不变）
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z, node_embeddings, num_nodes):
        # 解码潜在表示（完全不变）
        global_feat = self.latent_decoder(z)
        global_feat = global_feat.expand(num_nodes, -1)

        # 结合节点和全局特征（完全不变）
        combined = torch.cat([node_embeddings, global_feat], dim=-1)
        node_feat = self.node_refine(combined)  # [N, hidden_dim]

        # ===== 向量化的边预测（数学上和原版完全等价）=====

        # 创建所有节点对的拼接特征
        # feat_i: [N, 1, hidden_dim] -> [N, N, hidden_dim]
        # feat_j: [1, N, hidden_dim] -> [N, N, hidden_dim]
        feat_i = node_feat.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        feat_j = node_feat.unsqueeze(0).expand(num_nodes, num_nodes, -1)

        # 拼接: [N, N, hidden_dim*2]
        pair_features = torch.cat([feat_i, feat_j], dim=-1)

        # 展平: [N*N, hidden_dim*2]
        pair_features_flat = pair_features.view(-1, self.edge_mlp[0].in_features)

        # 批量通过MLP: [N*N, 1] -> [N*N]
        edge_probs_flat = self.edge_mlp(pair_features_flat).squeeze(-1)

        # 重塑回矩阵: [N, N]
        adj_recon = edge_probs_flat.view(num_nodes, num_nodes)

        # 确保对称性（原版通过 adj[i,j] = adj[j,i] 保证）
        adj_recon = (adj_recon + adj_recon.t()) / 2

        # ===== 改动结束 =====

        return adj_recon


# ====================== 其他部分完全不变 ======================

class GraphiTVAE(nn.Module):
    """GraphiT-VAE模型"""

    def __init__(self, node_features=9, hidden_dim=128, latent_dim=256,
                 num_layers=6, num_heads=8, dropout=0.1, max_nodes=50):
        super().__init__()

        self.encoder = GraphiTEncoder(
            node_features, hidden_dim, latent_dim,
            num_layers, num_heads, dropout, max_nodes
        )

        self.decoder = ImprovedDecoder(latent_dim, hidden_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, edge_index, num_nodes):
        # 编码: 拿到安全的初始特征 h_init
        h_init, mean, logvar = self.encoder(x, edge_index)

        # 重参数化
        z = self.reparameterize(mean, logvar)

        # 解码
        adj_recon = self.decoder(z, h_init, num_nodes)

        return adj_recon, mean, logvar, z


def stable_vae_loss(adj_recon, adj_target, mean, logvar, beta=1.0, free_bits=0.5):
    """稳定的VAE损失函数"""
    device = adj_recon.device

    # BCE损失（数值稳定）
    eps = 1e-8
    adj_recon = torch.clamp(adj_recon, eps, 1 - eps)

    # 动态权重平衡
    pos_weight = (adj_target == 0).sum() / (adj_target == 1).sum().clamp(min=1)
    pos_weight = torch.clamp(pos_weight, 1.0, 10.0)

    # 计算BCE
    bce_pos = -adj_target * torch.log(adj_recon) * pos_weight
    bce_neg = -(1 - adj_target) * torch.log(1 - adj_recon)
    bce_loss = (bce_pos + bce_neg).mean()

    # KL损失（带free bits）
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss = kl_loss / mean.size(0)

    # Free bits机制
    kl_loss = torch.max(kl_loss, torch.tensor(free_bits, device=device))

    # 总损失
    total_loss = bce_loss + beta * kl_loss

    return total_loss, bce_loss, kl_loss


class SimpleGraphDataset(Dataset):
    def __init__(self, pickle_file_path, max_nodes=50):
        self.max_nodes = max_nodes
        self.pickle_file = Path(pickle_file_path)

        if not self.pickle_file.exists():
            raise FileNotFoundError(f"未找到数据文件: {pickle_file_path}")

        logger.info(f"正在加载数据: {pickle_file_path}")
        self.load_data()

    def load_data(self):
        with open(self.pickle_file, 'rb') as f:
            raw_data = pickle.load(f)

        self.graphs = []
        failed_count = 0

        for item in raw_data:
            try:
                if not self._validate_data_item(item):
                    failed_count += 1
                    continue

                graph = self._create_graph_data(item)
                if graph is not None:
                    self.graphs.append(graph)
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1

        logger.info(f"成功加载 {len(self.graphs)} 个图，失败 {failed_count} 个")

    def _validate_data_item(self, item):
        required_keys = ['coords', 'aa_types', 'adjacency_matrix', 'num_nodes']

        for key in required_keys:
            if key not in item:
                return False

        coords = np.array(item['coords'])
        aa_types = item['aa_types']
        adj_matrix = np.array(item['adjacency_matrix'])

        if len(coords) != len(aa_types):
            return False
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            return False
        if adj_matrix.shape[0] != len(coords):
            return False
        if len(coords) > self.max_nodes or len(coords) == 0:
            return False

        return True

    def _create_graph_data(self, item):
        try:
            coords = np.array(item['coords'])
            aa_types = item['aa_types']
            adj_matrix = np.array(item['adjacency_matrix'])

            node_features = self._create_node_features(aa_types)
            edge_index = self._adj_to_edge_index(adj_matrix)

            graph_data = {
                'x': torch.FloatTensor(node_features),
                'edge_index': torch.LongTensor(edge_index),
                'coords': torch.FloatTensor(coords),
                'adj_matrix': torch.FloatTensor(adj_matrix),
                'pdb_id': item['pdb_id'],
                'num_nodes': len(coords),
                'num_edges': int(np.sum(adj_matrix) / 2)
            }

            return graph_data
        except Exception as e:
            return None

    def _create_node_features(self, aa_types):
        aa_classification = {
            'GLY': [1, 0, 0, 0, 0, 0, 0], 'ALA': [1, 0, 0, 0, 0, 0, 0],
            'VAL': [0, 1, 0, 0, 0, 0, 0], 'ILE': [0, 1, 0, 0, 0, 0, 0], 'LEU': [0, 1, 0, 0, 0, 0, 0],
            'PHE': [0, 0, 1, 0, 0, 0, 0], 'TRP': [0, 0, 1, 0, 0, 0, 0], 'TYR': [0, 0, 1, 0, 0, 0, 0],
            'SER': [0, 0, 0, 1, 0, 0, 0], 'THR': [0, 0, 0, 1, 0, 0, 0], 'CYS': [0, 0, 0, 1, 0, 0, 0],
            'MET': [0, 0, 0, 1, 0, 0, 0], 'ASN': [0, 0, 0, 1, 0, 0, 0], 'GLN': [0, 0, 0, 1, 0, 0, 0],
            'ASP': [0, 0, 0, 0, 1, 0, 0], 'GLU': [0, 0, 0, 0, 1, 0, 0],
            'LYS': [0, 0, 0, 0, 0, 1, 0], 'ARG': [0, 0, 0, 0, 0, 1, 0],
            'HIS': [0, 0, 0, 0, 0, 0, 1], 'PRO': [0, 0, 0, 0, 0, 0, 1],
        }

        h_donors = {'SER', 'THR', 'TYR', 'CYS', 'LYS', 'ARG', 'HIS', 'TRP', 'ASN', 'GLN'}
        h_acceptors = {'SER', 'THR', 'TYR', 'ASP', 'GLU', 'ASN', 'GLN', 'HIS'}

        features = []
        for aa in aa_types:
            clean_aa = aa.upper()[:3]
            aa_feat = aa_classification.get(clean_aa, [0, 0, 0, 0, 0, 0, 1])
            donor = [1 if clean_aa in h_donors else 0]
            acceptor = [1 if clean_aa in h_acceptors else 0]
            features.append(aa_feat + donor + acceptor)

        return np.array(features, dtype=np.float32)

    def _adj_to_edge_index(self, adj_matrix):
        if adj_matrix.size == 0:
            return np.array([[], []])

        edge_indices = np.where(adj_matrix > 0)
        return np.stack([edge_indices[0], edge_indices[1]], axis=0)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def simple_collate_fn(batch):
    return batch[0]


class GraphiTTrainer:
    def __init__(self, model, device='cpu', model_name="GraphiT-VAE"):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name

        self.metrics_history = {
            'train_losses': [],
            'val_losses': [],
            'train_roc': [],
            'val_roc': [],
            'train_pr_auc': [],
            'val_pr_auc': []
        }

    def train_epoch(self, dataloader, optimizer, epoch, total_epochs):
        self.model.train()
        total_loss = 0
        all_targets = []
        all_preds = []
        num_batches = 0

        # KL退火策略
        beta = min(1.0, epoch / (total_epochs * 0.5))

        print(f"\n开始训练第 {epoch + 1}/{total_epochs} 轮，Beta={beta:.3f}")
        start_time = time.time()

        for batch_idx, graph_data in enumerate(dataloader):
            x = graph_data['x'].to(self.device)
            edge_index = graph_data['edge_index'].to(self.device)
            adj_target = graph_data['adj_matrix'].to(self.device)
            num_nodes = graph_data['num_nodes']

            optimizer.zero_grad()

            # 前向传播
            adj_recon, mean, logvar, z = self.model(x, edge_index, num_nodes)

            # 计算损失
            loss, bce_loss, kl_loss = stable_vae_loss(
                adj_recon, adj_target, mean, logvar, beta, free_bits=0.3
            )

            # 检查数值稳定性
            if not torch.isfinite(loss):
                logger.warning(f"Loss not finite: {loss.item()}")
                continue

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 每10个batch输出一次进度
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                avg_loss = total_loss / num_batches
                print(
                    f"  批次 [{batch_idx + 1}/{len(dataloader)}] - Loss: {loss.item():.4f} (平均: {avg_loss:.4f}), BCE: {bce_loss.item():.4f}, KL: {kl_loss.item():.4f}")

            all_targets.append(adj_target.detach().cpu().numpy().flatten())
            all_preds.append(adj_recon.detach().cpu().numpy().flatten())

        # 计算指标
        if all_targets:
            all_targets = np.concatenate(all_targets)
            all_preds = np.concatenate(all_preds)

            if len(np.unique(all_targets)) > 1:
                train_roc = roc_auc_score(all_targets, all_preds)
                precision, recall, _ = precision_recall_curve(all_targets, all_preds)
                train_pr_auc = auc(recall, precision)
            else:
                train_roc = 0.0
                train_pr_auc = 0.0
        else:
            train_roc = 0.0
            train_pr_auc = 0.0

        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        print(
            f"  训练完成! 用时: {epoch_time:.2f}秒, 平均Loss: {avg_loss:.4f}, ROC: {train_roc:.4f}, PR-AUC: {train_pr_auc:.4f}")

        return avg_loss, train_roc, train_pr_auc

    def validate(self, dataloader, epoch, total_epochs):
        self.model.eval()
        total_loss = 0
        all_targets = []
        all_preds = []
        num_batches = 0

        beta = min(1.0, epoch / (total_epochs * 0.5))

        print(f"\n开始验证...")
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, graph_data in enumerate(dataloader):
                x = graph_data['x'].to(self.device)
                edge_index = graph_data['edge_index'].to(self.device)
                adj_target = graph_data['adj_matrix'].to(self.device)
                num_nodes = graph_data['num_nodes']

                adj_recon, mean, logvar, z = self.model(x, edge_index, num_nodes)
                loss, _, _ = stable_vae_loss(
                    adj_recon, adj_target, mean, logvar, beta, free_bits=0.3
                )

                if torch.isfinite(loss):
                    total_loss += loss.item()
                    num_batches += 1

                    all_targets.append(adj_target.cpu().numpy().flatten())
                    all_preds.append(adj_recon.cpu().numpy().flatten())

                # 每20个batch输出一次进度
                if (batch_idx + 1) % 20 == 0:
                    print(f"  验证批次 [{batch_idx + 1}/{len(dataloader)}]")

        # 计算指标
        if all_targets:
            all_targets = np.concatenate(all_targets)
            all_preds = np.concatenate(all_preds)

            if len(np.unique(all_targets)) > 1:
                val_roc = roc_auc_score(all_targets, all_preds)
                precision, recall, _ = precision_recall_curve(all_targets, all_preds)
                val_pr_auc = auc(recall, precision)
            else:
                val_roc = 0.0
                val_pr_auc = 0.0
        else:
            val_roc = 0.0
            val_pr_auc = 0.0

        val_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        print(
            f"  验证完成! 用时: {val_time:.2f}秒, 平均Loss: {avg_loss:.4f}, ROC: {val_roc:.4f}, PR-AUC: {val_pr_auc:.4f}")

        return avg_loss, val_roc, val_pr_auc

    def train_model(self, train_loader, val_loader, num_epochs=100, lr=0.0003):
        print("\n" + "=" * 80)
        print(f"开始训练 {self.model_name}")
        print(f"设备: {self.device}")
        print(f"训练集大小: {len(train_loader)}, 验证集大小: {len(val_loader)}")
        print(f"总轮数: {num_epochs}, 学习率: {lr}")
        print("=" * 80)

        # 优化器设置
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 学习率调度
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=10
        )

        best_val_roc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\n{'=' * 60}")
            # 修复：直接从optimizer获取学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"轮次 {epoch + 1}/{num_epochs} - 学习率: {current_lr:.6f}")
            print(f"{'=' * 60}")

            # 训练
            train_loss, train_roc, train_pr_auc = self.train_epoch(
                train_loader, optimizer, epoch, num_epochs
            )

            # 验证
            val_loss, val_roc, val_pr_auc = self.validate(
                val_loader, epoch, num_epochs
            )

            # 修复：传入验证损失
            scheduler.step(val_loss)

            # 记录
            self.metrics_history['train_losses'].append(train_loss)
            self.metrics_history['val_losses'].append(val_loss)
            self.metrics_history['train_roc'].append(train_roc)
            self.metrics_history['val_roc'].append(val_roc)
            self.metrics_history['train_pr_auc'].append(train_pr_auc)
            self.metrics_history['val_pr_auc'].append(val_pr_auc)

            # 保存最佳模型
            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"\n>>> 新的最佳模型! ROC: {best_val_roc:.4f} <<<")
            else:
                patience_counter += 1
                print(f"  未改进，耐心计数: {patience_counter}/30")

            # 早停
            if patience_counter > 30:
                print(f"\n>>> 早停于第 {epoch + 1} 轮 <<<")
                break

            # 轮次总结
            print(f"\n轮次总结:")
            print(f"  训练 - Loss: {train_loss:.4f}, ROC: {train_roc:.4f}, PR-AUC: {train_pr_auc:.4f}")
            print(f"  验证 - Loss: {val_loss:.4f}, ROC: {val_roc:.4f}, PR-AUC: {val_pr_auc:.4f}")
            print(f"  最佳验证ROC: {best_val_roc:.4f}")

        # 加载最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\n已加载最佳模型 (ROC: {best_val_roc:.4f})")

        return best_val_roc


def train_graphit_vae(pickle_file_path, num_epochs=100, max_nodes=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 80}")
    print(f"GraphiT-VAE 训练准备（最小改动优化版）")
    print(f"{'=' * 80}")
    print(f"使用设备: {device}")
    print(f"数据文件: {pickle_file_path}")

    # 加载数据
    print("\n正在加载数据集...")
    dataset = SimpleGraphDataset(pickle_file_path, max_nodes=max_nodes)
    print(f"数据集加载完成! 总共 {len(dataset)} 个图")

    # 划分数据集
    print("\n正在划分数据集...")
    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_dataset)} 个图")
    print(f"  验证集: {len(val_dataset)} 个图")
    print(f"  测试集: {len(test_dataset)} 个图")

    # 数据加载器（小改动：增加num_workers和pin_memory）
    print("\n创建数据加载器...")
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=simple_collate_fn,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=simple_collate_fn,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=simple_collate_fn
    )

    # 创建模型
    print("\n正在创建GraphiT-VAE模型...")
    print(f"  节点特征维度: 9")
    print(f"  隐藏层维度: 128")
    print(f"  潜在空间维度: 256")
    print(f"  层数: 6")
    print(f"  注意力头数: 8")

    model = GraphiTVAE(
        node_features=9,
        hidden_dim=128,
        latent_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        max_nodes=max_nodes
    )

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 训练
    print("\n" + "=" * 80)
    print("开始训练!")
    print("=" * 80)

    trainer = GraphiTTrainer(model, device)
    best_val_roc = trainer.train_model(train_loader, val_loader, num_epochs=num_epochs, lr=0.0003)

    # 测试
    print("\n" + "=" * 80)
    print("开始测试...")
    print("=" * 80)

    test_loss, test_roc, test_pr_auc = trainer.validate(test_loader, num_epochs, num_epochs)

    print("\n" + "=" * 80)
    print(f"GraphiT-VAE 最终结果:")
    print(f"  最佳验证ROC: {best_val_roc:.4f}")
    print(f"  测试ROC: {test_roc:.4f}")
    print(f"  测试PR-AUC: {test_pr_auc:.4f}")
    print("=" * 80)

    # 在train_graphit_vae函数的最后
    torch.save(trainer.model.state_dict(), 'results/graphit_vae_best.pth')
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'metrics_history': trainer.metrics_history,
        'test_roc': test_roc,
        'test_pr_auc': test_pr_auc
    }, 'results/graphit_vae_checkpoint.pth')

    return trainer, test_roc


if __name__ == "__main__":
    pickle_file_path = "D:\\pythonstu\\pythonProject16\\dev\\new_gat-vae\\final_results\\final_active_sites_with_graphs_full_data.pkl"

    trainer, test_roc = train_graphit_vae(
        pickle_file_path,
        num_epochs=100,
        max_nodes=50
    )

    print(f"\nGraphiT-VAE 测试ROC: {test_roc:.4f}")