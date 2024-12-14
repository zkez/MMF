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


def probability_flow_sampling(score_model, cond_emb, device, steps=50, eps=1e-5, shape=(1,1,256,256)):
    with torch.no_grad():
        t_vals = torch.linspace(1.0, eps, steps=steps, device=device)
        x = sigma_max * torch.randn(shape, device=device)
        
        for i in range(steps-1):
            t = t_vals[i].unsqueeze(0)
            t_emb = score_model.time_embed(t)
            cond_out = score_model.cond_fc(cond_emb)
            
            # score = Φθ(p(t), t|O)
            score = score_model(x, t_emb, cond_out)
            

            dt = t_vals[i+1]-t_vals[i]
            dsigma_sq_dt = torch.tensor(d_sigma_sq_dt(t.item()), device=device, dtype=torch.float32)
            sig_t = sigma_t(t.item())
            d_sigma_dt = dsigma_sq_dt/(2*sig_t)
            dp_dt = -sig_t * d_sigma_dt * score
            x = x + dp_dt * dt
            
        return x


def rank_and_filter_candidates(energy_model, candidates, cond_emb, device, eps=1e-5, prune_ratio=0.2):
    with torch.no_grad():
        t = torch.full((candidates.size(0),1), eps, device=device)
        t_emb = energy_model.time_embed(t)
        cond_out = energy_model.cond_fc(cond_emb)
        
        # Φϕ(p, ε|O)
        phi_out = energy_model(candidates, t_emb, cond_out)
        # Energy = <p,Φϕ>
        B = candidates.size(0)
        p_flat = candidates.view(B,-1)
        phi_flat = phi_out.view(B,-1)
        energy = torch.sum(p_flat * phi_flat, dim=1)
        
        # 根据energy排序
        sorted_energy, indices = torch.sort(energy, descending=True)
        keep_num = int(B * (1 - prune_ratio))
        top_indices = indices[:keep_num]
        
        top_candidates = candidates[top_indices]
        top_energies = sorted_energy[:keep_num]
        
        weights = F.softmax(top_energies, dim=0).view(-1,1,1,1)
        final = torch.sum(top_candidates * weights, dim=0, keepdim=True)
        
        return final, top_candidates, top_energies
    