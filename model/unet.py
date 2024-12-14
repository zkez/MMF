import torch
import torch.nn as nn


class UNetBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)


class UNetModel3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super(UNetModel3D, self).__init__()
        self.down1 = UNetBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = UNetBlock3D(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.mid = UNetBlock3D(base_channels*2, base_channels*2)
        
        # 上采样部分匹配通道数
        self.up2 = nn.ConvTranspose3d(base_channels*2, base_channels*2, kernel_size=2, stride=2)
        self.up_block2 = UNetBlock3D(base_channels*4, base_channels*2)
        
        self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.up_block1 = UNetBlock3D(base_channels*2, base_channels)
        
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        d1 = self.down1(x)      # [B,base_channels,D,H,W]
        p1 = self.pool1(d1)     # [B,base_channels,D/2,H/2,W/2]
        
        d2 = self.down2(p1)     # [B,2*base_channels,D/2,H/2,W/2]
        p2 = self.pool2(d2)     # [B,2*base_channels,D/4,H/4,W/4]
        
        m = self.mid(p2)        # [B,2*base_channels,D/4,H/4,W/4]
        
        u2 = self.up2(m)        # [B,2*base_channels,D/2,H/2,W/2]
        cat2 = torch.cat([u2, d2], dim=1) # [B,4*base_channels,D/2,H/2,W/2]
        u2_block = self.up_block2(cat2)   # [B,2*base_channels,D/2,H/2,W/2]
        
        u1 = self.up1(u2_block) # [B,base_channels,D,H,W]
        cat1 = torch.cat([u1, d1], dim=1) # [B,2*base_channels,D,H,W]
        u1_block = self.up_block1(cat1)   # [B,base_channels,D,H,W]
        
        out = self.final_conv(u1_block)   # [B,out_channels,D,H,W]

        return out
    