"""
Transformer SELFIES 论文专用版 v5.0
✅ 继承 v4.0 所有内核 (Attention Pooling, Augmentation, Low Dropout)
✅ 新增: 实验数据记录器 (CSV Logger) -> 方便后续画 Loss/Acc 曲线
✅ 新增: 实时准确率监控 (Real-time Accuracy Check)
✅ 新增: 双重模型保存 (Best Loss & Best Accuracy)
"""
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from rdkit import Chem
import math
import selfies as sf
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import argparse
import random
import pandas as pd # 需要安装 pandas
import csv
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================================================================
# Part 1: Tokenizer (保持不变)
# ===================================================================
class SELFIESTokenizer:
    def __init__(self, vocab_list=None):
        self.special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        if vocab_list is not None:
            self.vocab = self.special_tokens.copy()
            for i, token in enumerate(vocab_list):
                if token not in self.special_tokens:
                    self.vocab[token] = i + len(self.special_tokens)
            self.is_frozen = True
        else:
            self.vocab = self.special_tokens.copy()
            self.is_frozen = False
        self.idx2token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def add_token(self, token):
        if self.is_frozen: return False
        if token not in self.vocab:
            self.vocab[token] = self.vocab_size
            self.idx2token[self.vocab_size] = token
            self.vocab_size += 1
            return True
        return False

    def tokenize(self, selfies_str):
        try:
            tokens = list(sf.split_selfies(selfies_str))
            if not self.is_frozen:
                for token in tokens: self.add_token(token)
            return tokens
        except: return ['<unk>']

    def encode(self, selfies_str, max_length=200):
        tokens = ['<sos>'] + self.tokenize(selfies_str) + ['<eos>']
        if len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + ['<eos>']
        else:
            tokens.extend(['<pad>'] * (max_length - len(tokens)))
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

    def decode(self, indices):
        tokens = []
        for idx in indices:
            token = self.idx2token.get(int(idx), '<unk>')
            if token == '<eos>': break
            elif token not in ['<pad>', '<sos>', '<unk>']: tokens.append(token)
        return ''.join(tokens)

    def get_vocab_list(self):
        return [self.idx2token[idx] for idx in range(len(self.special_tokens), self.vocab_size) if idx in self.idx2token]

# ===================================================================
# Part 2: 模型核心 (v4.0 内核)
# ===================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300, dropout=0.05):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerSELFIESAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.05,
                 latent_dim=256, max_length=300):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_length, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Attention Pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(),
            nn.Linear(d_model // 2, 1), nn.Softmax(dim=1)
        )

        self.to_latent = nn.Sequential(nn.Linear(d_model, latent_dim), nn.Tanh())
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, d_model), nn.GELU())

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p, gain=0.01)

    def encode(self, src, src_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
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

    def decode(self, latent, tgt=None, max_steps=100):
        device = latent.device
        memory = self.from_latent(latent).unsqueeze(1)
        if tgt is not None:
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            sz = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
            output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            return self.output(output)
        else:
            return self._greedy_decode_logits(memory, max_steps, device)

    def _greedy_decode_logits(self, memory, max_steps, device):
        batch_size = memory.size(0)
        input_seq = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        logits_list = []
        for _ in range(max_steps):
            tgt_emb = self.embedding(input_seq) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            tgt_mask = torch.triu(torch.ones(input_seq.size(1), input_seq.size(1), device=device), diagonal=1)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
            out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            step_logits = self.output(out[:, -1:, :])
            logits_list.append(step_logits)
            next_token = step_logits.argmax(dim=-1)
            next_token[finished] = 0
            input_seq = torch.cat([input_seq, next_token], dim=1)
            finished |= (next_token.squeeze(-1) == 2)
            if finished.all(): break
        return torch.cat(logits_list, dim=1)

    def forward(self, src, tgt=None):
        if src.dim() == 1: src = src.unsqueeze(0)
        src_mask = (src == 0)
        latent = self.encode(src, src_mask)
        if tgt is not None:
            output = self.decode(latent, tgt[:, :-1])
            return output, latent
        else:
            output = self.decode(latent, None)
            return output, latent

# ===================================================================
# Part 3: 数据加载
# ===================================================================
class ChEMBLDataLoader:
    def __init__(self, data_path, max_molecules=200000):
        self.data_path = data_path
        self.max_molecules = max_molecules

    def scan_and_build_vocab(self):
        logger.info(f"🔍 扫描数据文件: {self.data_path}")
        temp_tokenizer = SELFIESTokenizer()
        with open(self.data_path, 'r', encoding='utf-8') as f:
            smiles_list = []
            for line in f:
                s = line.strip().split()[0]
                if len(s) > 1: smiles_list.append(s)
                if len(smiles_list) >= self.max_molecules * 2: break

        valid_selfies = []
        for s in tqdm(smiles_list, desc="构建词汇表"):
            try:
                if Chem.MolFromSmiles(s):
                    se = sf.encoder(s)
                    if se and 10 < len(se) < 300:
                        valid_selfies.append(se)
                        temp_tokenizer.tokenize(se)
                        if len(valid_selfies) >= self.max_molecules: break
            except: pass
        return valid_selfies, temp_tokenizer.get_vocab_list()

    def load_data_with_fixed_vocab(self, vocab_list):
        logger.info(f"📚 加载数据...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            smiles_list = []
            for line in f:
                s = line.strip().split()[0]
                if len(s) > 1: smiles_list.append(s)
                if len(smiles_list) >= self.max_molecules: break
        tokenizer = SELFIESTokenizer(vocab_list=vocab_list)
        return smiles_list, tokenizer

class AugmentedSELFIESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=300, augment=True):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        if self.augment and random.random() > 0.5:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol: smi = Chem.MolToSmiles(mol, doRandom=True)
            except: pass
        try:
            selfies_str = sf.encoder(smi)
            if not selfies_str: selfies_str = "[C]"
        except: selfies_str = "[C]"
        encoded = self.tokenizer.encode(selfies_str, self.max_length)
        return torch.tensor(encoded, dtype=torch.long)

# ===================================================================
# Part 4: 实验记录器 (🔥 新增)
# ===================================================================
class ExperimentLogger:
    def __init__(self, log_file="training_log.csv"):
        self.log_file = log_file
        # 初始化 CSV
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Accuracy', 'LR', 'Time_Sec'])

    def log(self, epoch, train_loss, val_loss, val_acc, lr, time_sec):
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_acc, lr, time_sec])
        # 顺便打个日志到控制台
        logger.info(f"📊 Epoch {epoch} Report: T-Loss={train_loss:.4f}, V-Loss={val_loss:.4f}, Acc={val_acc:.2f}%, LR={lr:.6f}")

def check_accuracy(model, val_loader, tokenizer, device, num_check=200):
    """抽样检测准确率"""
    model.eval()
    correct = 0
    total = 0

    # 随机取几个 batch 检查，不要检查全部以免太慢
    check_iter = iter(val_loader)

    with torch.no_grad():
        for _ in range(5): # 检查约 5 个 batch
            try:
                batch = next(check_iter).to(device)
            except StopIteration:
                break

            latent = model.encode(batch, (batch==0))
            logits = model.decode(latent, None, max_steps=150) # 给够长度
            preds = logits.argmax(dim=-1)

            for i in range(len(preds)):
                if total >= num_check: break

                true_str = tokenizer.decode(batch[i].cpu().numpy())
                pred_str = tokenizer.decode(preds[i].cpu().numpy())

                # 简单字符串对比即可，作为训练监控足够了
                if true_str == pred_str:
                    correct += 1
                total += 1
            if total >= num_check: break

    return (correct / total * 100) if total > 0 else 0

# ===================================================================
# Part 5: 训练函数 (集成 Logger)
# ===================================================================
def train_paper_version(model, train_loader, val_loader, epochs, device, tokenizer, output_path, start_epoch=0):
    logger.info("🚀 启动 v5.0 论文专用训练 (带 Logger & Acc Check)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.1
    )
    scaler = GradScaler()

    # 初始化记录器
    exp_logger = ExperimentLogger(log_file="results/training_history.csv")

    best_val_loss = float('inf')
    best_accuracy = 0.0

    Path('results').mkdir(exist_ok=True)

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", postfix={'loss': 0.0})

        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                output, latent = model(batch, batch)
                target = batch[:, 1:]
                loss = F.cross_entropy(output.reshape(-1, output.size(-1)), target.reshape(-1), ignore_index=0, label_smoothing=0.1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                with autocast():
                    output, latent = model(batch, batch)
                    loss = F.cross_entropy(output.reshape(-1, output.size(-1)), batch[:, 1:].reshape(-1), ignore_index=0)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # 🔥 实时检查准确率
        current_acc = check_accuracy(model, val_loader, tokenizer, device)

        # 记录日志
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        exp_logger.log(epoch, train_loss, val_loss, current_acc, current_lr, epoch_time)

        # 保存策略1: Best Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'vocab_list': tokenizer.get_vocab_list(),
                'config': {
                    'vocab_size': tokenizer.vocab_size, 'd_model': model.d_model,
                    'nhead': 8, 'num_encoder_layers': 6, 'num_decoder_layers': 6,
                    'dim_feedforward': 2048, 'latent_dim': model.latent_dim,
                    'max_length': model.max_length, 'dropout': 0.05
                }
            }, output_path)
            logger.info(f"💾 模型保存 (Best Loss: {val_loss:.4f})")

        # 保存策略2: Best Accuracy (另存一份)
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            best_acc_path = output_path.replace(".pth", "_best_acc.pth")
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'vocab_list': tokenizer.get_vocab_list(),
                'config': {
                    'vocab_size': tokenizer.vocab_size, 'd_model': model.d_model,
                    'nhead': 8, 'num_encoder_layers': 6, 'num_decoder_layers': 6,
                    'dim_feedforward': 2048, 'latent_dim': model.latent_dim,
                    'max_length': model.max_length, 'dropout': 0.05
                }
            }, best_acc_path)
            logger.info(f"🌟 新高准确率! 已另存为 {best_acc_path}")

# ===================================================================
# Part 6: 主程序
# ===================================================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'/root/autodl-tmp/pythonProject16/dev/data/chembl_smiles.txt')
    parser.add_argument('--output_path', type=str, default='results/best_transformer_v4.pth') # 改名防止覆盖
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_molecules', type=int, default=250000)
    parser.add_argument('--epochs', type=int, default=100)
    # v4增强版因为计算量大，建议 128 batch size
    parser.add_argument('--batch_size', type=int, default=256)
    # 建议设置 max_length=200 兼顾速度和覆盖率，300 也可以
    parser.add_argument('--max_length', type=int, default=300)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("="*60)
    logger.info(f"🚀 启动 v5.0 论文专用训练 | Data: {args.max_molecules} | MaxLen: {args.max_length}")
    logger.info("="*60)

    loader = ChEMBLDataLoader(args.data_path, max_molecules=args.max_molecules)
    _, vocab_list = loader.scan_and_build_vocab()
    smiles_list, tokenizer = loader.load_data_with_fixed_vocab(vocab_list)

    train_size = int(0.9 * len(smiles_list))
    train_dataset = AugmentedSELFIESDataset(smiles_list[:train_size], tokenizer, max_length=args.max_length, augment=True)
    val_dataset = AugmentedSELFIESDataset(smiles_list[train_size:], tokenizer, max_length=args.max_length, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = TransformerSELFIESAutoencoder(
        vocab_size=tokenizer.vocab_size, d_model=512, nhead=8,
        num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
        dropout=0.05,
        latent_dim=256, max_length=args.max_length
    ).to(device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1

    train_paper_version(model, train_loader, val_loader, args.epochs, device, tokenizer, args.output_path, start_epoch)