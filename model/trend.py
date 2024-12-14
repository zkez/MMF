import torch
import torch.nn as nn

class TrendExtractor3D(nn.Module):
    def __init__(self, in_channels=2, embed_dim=64):
        super(TrendExtractor3D, self).__init__()
        # 将(ct_t0, ct_t2)拼接后通过3D卷积提取全局特征
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.fc = nn.Linear(32, embed_dim)
        
    def forward(self, T0, T2):
        # T0, T2: [B, 1, D, H, W]
        x = torch.cat([T0, T2], dim=1) # [B, 2, D, H, W]
        feat = self.conv(x) # [B, 32, 1, 1, 1]
        feat = feat.view(feat.size(0), -1) # [B, 32]
        trend = self.fc(feat) # [B, 64]
        
        return trend
    