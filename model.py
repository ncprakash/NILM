import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedSeq2PointBackbone(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        specs = [
            (1,  32, 5, 1),
            (32, 32, 5, 2),
            (32, 40, 5, 4),
            (40, 50, 5, 8),
            (50, 50, 5, 16),
        ]
        self.layers = nn.ModuleList()
        for in_ch, out_ch, k, d in specs:
            pad = ((k - 1) * d) // 2
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MidpointAttention(nn.Module):
    def __init__(self, d_model, n_heads=5, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features):
        x = features.transpose(1, 2)
        midpoint_idx = x.shape[1] // 2
        query = x[:, midpoint_idx:midpoint_idx+1, :]
        attn_out, _ = self.attn(query, x, x)
        attn_out = self.norm(attn_out + query)
        return attn_out.squeeze(1)


class DilatedSeq2Point(nn.Module):
    def __init__(self, dropout_cnn=0.1, dropout_fc=0.3, n_heads=5):
        super().__init__()
        self.backbone = DilatedSeq2PointBackbone(dropout=dropout_cnn)
        self.attention = MidpointAttention(d_model=50, n_heads=n_heads, dropout=dropout_cnn)
        self.fc1 = nn.Linear(50, 512)
        self.dropout = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        feat = self.backbone(x)
        ctx = self.attention(feat)
        h = F.gelu(self.fc1(ctx))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)
