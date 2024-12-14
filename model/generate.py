import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('../')
from model.unet import UNetModel3D

sigma_min = 0.01
sigma_max = 50.0


class TimeEmbedding(nn.Module):
    """
    Time embedding layer    
    """
    def __init__(self, embed_dim=64):
        super(TimeEmbedding, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.fc(x)
    

class ScoreModel3D(nn.Module):
    def __init__(self, in_channels=1, cond_dim=64, time_dim=64, base_channels=32):
        super(ScoreModel3D, self).__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.cond_fc = nn.Linear(cond_dim, time_dim)
        
        # 使用3D UNet作为基础网络
        # 输出与输入维度相同，用于估计score
        self.unet = UNetModel3D(in_channels, in_channels, base_channels=base_channels)
        
    def forward(self, x, t_emb, cond_emb):
        # x: [B,1,D,H,W]
        # t_emb: [B,time_dim]
        # cond_emb: [B,time_dim]
        
        # 简化处理，将t_emb和cond_emb相加作为全局条件，不对UNet结构额外处理
        # 实际中可在UNet中注入条件（例如使用FiLM、AdaIN等），这里简化
        # out: [B,1,D,H,W]
        out = self.unet(x)

        return out


class EnergyModel3D(nn.Module):
    def __init__(self, in_channels=1, cond_dim=64, time_dim=64, base_channels=32):
        super(EnergyModel3D, self).__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.cond_fc = nn.Linear(cond_dim, time_dim)
        
        self.unet = UNetModel3D(in_channels, in_channels, base_channels=base_channels)

    def forward(self, x, t_emb, cond_emb):
        # 返回Φϕ(p,t|O)
        # x: [B,1,D,H,W]
        feat = self.unet(x)  # [B,1,D,H,W]

        return feat
    

def sigma_t(t):
    return sigma_min * (sigma_max / sigma_min)**t


def d_sigma_sq_dt(t):
    log_ratio = np.log(sigma_max/sigma_min)

    sigma_t_val = sigma_t(t)
    d_sigma_dt = sigma_t_val * log_ratio

    return 2 * sigma_t_val * d_sigma_dt


def dsm_loss_score_model(score_model, x_0, cond_emb, device):
    """
    Compute DSM loss for score model

    params:
    - x_0: 
    - cond_emb: 
    """
    epsilon = 1e-5
    t = torch.rand(x_0.size(0), device=device)*(1.0 - epsilon) + epsilon
    t = t.unsqueeze(1)

    sigma = sigma_t(t)
    noise = torch.randn_like(x_0)
    x_t = x_0 + sigma * noise

    t_emb = score_model.time_embedding(t)
    cond_out = score_model.cond_fc(cond_emb)
    score_pred = score_model(x_t, t_emb, cond_out)

    target = (x_0 - x_t) / sigma
    loss = F.mse_loss(score_pred, target, reduction='mean')

    return loss


def dsm_loss_energy_model(energy_model, x_0, cond_emb, device):
    """
    Compute DSM loss for energy model

    params:
    - x_0:
    - cond_emb:
    """
    epsilon = 1e-5
    t = torch.rand(x_0.size(0), device=device)*(1.0 - epsilon) + epsilon
    t = t.unsqueeze(1)
    sigma = sigma_t(t)
    noise = torch.randn_like(x_0)
    x_t = x_0 + sigma * noise
    
    t_emb = energy_model.time_embedding(t)
    cond_out = energy_model.cond_fc(cond_emb)
    phi_out = energy_model(x_t, t_emb, cond_out)
    
    target = (x_0 - x_t) / sigma
    loss = F.mse_loss(phi_out, target, reduction='mean')

    return loss
 