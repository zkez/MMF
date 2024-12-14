import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append('../')
from model.generate import sigma_t


class EnergyTrainer:
    def __init__(self, energy_model, trend_extractor, device='cuda'):
        self.energy_model = energy_model.to(device)
        self.trend_extractor = trend_extractor.to(device)
        self.device = device

    def dsm_loss(self, T0, T1, T2):
        B = T1.size(0)
        epsilon = 1e-5
        # 随机采样 t ~ U(epsilon, 1)
        t = torch.rand(B, device=self.device)*(1.0 - epsilon) + epsilon
        t = t.unsqueeze(1) # [B,1]

        # 提取条件特征
        cond_emb = self.trend_extractor(T0, T2) # [B,64]
        t_emb = self.energy_model.time_embed(t) # [B,64]
        cond_out = self.energy_model.cond_fc(cond_emb) # [B,64]

        # sigma(t)
        sigma = sigma_t(t)
        sigma = sigma.view(B,1,1,1,1) # [B,1,1,1,1]

        # 加噪
        noise = torch.randn_like(T1)
        T1_noisy = T1 + sigma * noise # [B,1,D,H,W]

        # Energy model输出Φϕ(p,t|O)
        phi_out = self.energy_model(T1_noisy, t_emb, cond_out) # [B,1,D,H,W]

        # 真实的Φϕ = score = noise
        target = noise

        # DSM损失
        loss = torch.mean((phi_out - target)**2)

        return loss


def train_energy_model(energy_model, trend_extractor, train_dataset, epochs=10, batch_size=2, lr=1e-4, device='cuda'):
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    trainer = EnergyTrainer(energy_model, trend_extractor, device=device)
    optimizer = optim.Adam(list(energy_model.parameters()) + list(trend_extractor.parameters()), lr=lr)

    for epoch in range(epochs):
        energy_model.train()
        trend_extractor.train()
        total_loss = 0.0
        for T0, T1, T2, _ in dataloader:
            T0, T1, T2 = T0.to(device), T1.to(device), T2.to(device)
            optimizer.zero_grad()
            loss = trainer.dsm_loss(T0, T1, T2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[EnergyModel] Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")