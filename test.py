import torch

import sys
sys.path.append('../')
from model.generate import ScoreModel3D, EnergyModel3D, dsm_loss_score_model, dsm_loss_energy_model, probability_flow_sampling, rank_and_filter_candidates
from model.trend import TrendExtractor3D


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B, C, D, H, W = 4, 1, 16, 64, 64
    ct_t0 = torch.randn(B,1,D,H,W,device=device)
    ct_t2 = torch.randn(B,1,D,H,W,device=device)
    M_data = torch.randn(B,1,D,H,W,device=device)
    trend_extractor = TrendExtractor3D(in_channels=3, embed_dim=64).to(device)
    cond_emb = trend_extractor(ct_t0, ct_t2, M_data)  # [B,64]
    
    score_model = ScoreModel3D(in_channels=1, cond_dim=64, time_dim=64, base_channels=32).to(device)
    energy_model = EnergyModel3D(in_channels=1, cond_dim=64, time_dim=64, base_channels=32).to(device)
    
    t = torch.rand(B,1,device=device) # [B,1]
    t_emb = score_model.time_embed(t)
    cond_out = score_model.cond_fc(cond_emb) # [B,64]
    
    x = torch.randn(B,1,D,H,W,device=device)
    score_out = score_model(x, t_emb, cond_out)
    print("ScoreModel3D output:", score_out.shape) # expect [B,1,D,H,W]

    # energy model
    t_emb_e = energy_model.time_embed(t)
    cond_out_e = energy_model.cond_fc(cond_emb)
    phi_out = energy_model(x, t_emb_e, cond_out_e)
    print("EnergyModel3D output:", phi_out.shape) # [B,1,D,H,W]
    