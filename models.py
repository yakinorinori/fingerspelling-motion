"""
比較する2つのモデル（入力 (B,T,D)、mask (B,T) は本物のフレームが True）

  CNN1D       : dilated 1D-CNN（k=3, dilation 1,2,4,8 → 受容野31フレーム）
  Transformer : TransformerModel.py の TimeSeriesTransformer を元にしたもの
                変更点：入力を d_model 次元に線形変換（特徴量の次元数が nhead で割り切れなくてもよい）、
                        パディングをマスクする、batch_first
どちらも最後は「本物のフレームだけの平均」→ 全結合でクラスを出す。
"""
import math
import os

# Mac(MPS)で未対応の処理があっても CPU で代わりに計算させる（torch を読み込む前に設定する）
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch import nn


def masked_mean(x, mask):
    """x (B,T,C), mask (B,T) → (B,C)"""
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class CNN1D(nn.Module):
    def __init__(self, in_dim, num_classes, channels=64, kernel_size=3, dilations=(1, 2, 4, 8), dropout=0.2):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, channels, 1)
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=d * (kernel_size - 1) // 2, dilation=d),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        self.head = nn.Linear(channels, num_classes)
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, x, mask):
        m = mask.unsqueeze(1).float()
        h = self.inp(x.transpose(1, 2).contiguous()) * m   # (B,C,T)。Mac(MPS)対策で contiguous にする
        for block in self.blocks:
            h = ((h + block(h)) * m).contiguous()     # 残差接続、パディング部分は0に戻す
        return self.head(masked_mean(h.transpose(1, 2).contiguous(), mask))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Transformer(nn.Module):
    def __init__(self, in_dim, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        # enable_nested_tensor=False：Mac(MPS)で未対応の処理を使わないようにする
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask):
        h = self.drop(self.pos(self.inp(x)))
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.head(masked_mean(h, mask))


MODELS = {"cnn": CNN1D, "transformer": Transformer}


def build_model(name, in_dim, num_classes):
    return MODELS[name](in_dim, num_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
