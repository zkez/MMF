import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

device = 'cuda'
sigma_min = 0.01
sigma_max = 50.0
epsilon = 1e-5


def sigma_t(t):
    return sigma_min * (sigma_max / sigma_min)**t

def d_sigma_sq_dt(t):
    # 这里需要根据VE-SDE定义的公式实现
    # dp = sqrt(d[sigma^2(t)]/dt)*dw
    # sigma(t)=sigma_min*(sigma_max/sigma_min)^t
    # dsigma/dt = sigma(t)*ln(sigma_max/sigma_min)
    import math
    log_ratio = math.log(sigma_max/sigma_min)
    sig = sigma_t(t)
    d_sigma_dt = sig * log_ratio
    return 2*sig*d_sigma_dt


class ProbabilityFlowODEFunc3D(nn.Module):
    def __init__(self, score_model, cond_emb, sigma_t_func, d_sigma_sq_dt_func):
        super(ProbabilityFlowODEFunc3D, self).__init__()
        self.score_model = score_model
        self.cond_emb = cond_emb
        self.sigma_t_func = sigma_t_func
        self.d_sigma_sq_dt_func = d_sigma_sq_dt_func

    def forward(self, t, x):
        # x: [B, C, D, H, W]
        # t: scalar tensor
        t_ = t.reshape(1,1) # [1,1]
        t_emb = self.score_model.time_embed(t_) # [1,64]
        cond_out = self.score_model.cond_fc(self.cond_emb) # [B,64]
        # Broadcast t_emb and cond_out to match batch size
        # Assuming cond_emb is [B,64] and t_emb is [1,64], broadcast to [B,64]
        t_emb = t_emb.expand(x.size(0), -1) # [B,64]
        cond_out = cond_out # [B,64]

        # 预测score
        score = self.score_model(x, t_emb, cond_out) # [B,1,D,H,W]

        # 计算 dp/dt = -sigma(t) * d_sigma/dt * score
        t_val = t.item()
        sigma = self.sigma_t_func(t_val)
        dsigma_sq = self.d_sigma_sq_dt_func(t_val)
        d_sigma = dsigma_sq / (2 * sigma)
        dp_dt = -sigma * d_sigma * score # [B,1,D,H,W]

        return dp_dt


def probability_flow_sampling(score_model, cond_emb, sigma_t_func, d_sigma_sq_dt_func, sigma_max, eps=1e-5, shape=(1,1,16,64,64), device='cuda'):
    # 初始化初始条件 p(1) ~ N(0, sigma_max^2 I)
    x_init = sigma_max * torch.randn(shape, device=device)

    # 定义ODE函数
    ode_func = ProbabilityFlowODEFunc3D(score_model, cond_emb, sigma_t_func, d_sigma_sq_dt_func).to(device)

    # 时间区间 [1.0, eps]
    t_span = torch.tensor([1.0, eps], device=device, dtype=torch.float32)

    # 使用odeint求解概率流ODE
    solution = odeint(ode_func, x_init, t_span, method='dopri5', atol=1e-5, rtol=1e-5)

    # solution.shape = [2, B, C, D, H, W]
    # solution[0]对应t=1.0的状态 (即初始值), solution[-1]对应t=eps时的解
    final_x = solution[-1] # [B, C, D, H, W]
    return final_x


def rank_and_filter_candidates(energy_model, cond_emb, candidates, prune_ratio=0.2):
    """
    energy_model: 已训练的 EnergyModel3D
    cond_emb: [B, 64]
    candidates: [N, C, D, H, W]
    prune_ratio: 剔除的低分比例
    """
    device = candidates.device
    N = candidates.size(0)
    B = cond_emb.size(0)
    # 为每个候选重复 cond_emb
    cond_emb_expanded = cond_emb.repeat(N, 1) # [N, 64]

    # 定义时间 t = eps
    t = torch.full((N,1), epsilon, device=device, dtype=torch.float32) # [N,1]

    # 计算 t_emb 和 cond_out
    t_emb = energy_model.time_embed(t) # [N,64]
    cond_out = energy_model.cond_fc(cond_emb_expanded) # [N,64]

    # 计算 Φϕ(p, t|O)
    phi_out = energy_model(candidates, t_emb, cond_out) # [N,1,D,H,W]

    # 计算能量 E = <p, Φϕ>
    p_flat = candidates.view(N, -1) # [N, C*D*H*W]
    phi_flat = phi_out.view(N, -1) # [N, C*D*H*W]
    energy = torch.sum(p_flat * phi_flat, dim=1) # [N]

    # 根据能量排序
    sorted_energy, indices = torch.sort(energy, descending=True)
    keep_num = int(N * (1 - prune_ratio))
    top_indices = indices[:keep_num]
    top_candidates = candidates[top_indices] # [keep_num, C, D, H, W]
    top_energies = sorted_energy[:keep_num] # [keep_num]

    # 基于能量的加权平均
    weights = torch.softmax(top_energies, dim=0).view(-1,1,1,1,1) # [keep_num,1,1,1,1]
    final = torch.sum(top_candidates * weights, dim=0, keepdim=True) # [1, C, D, H, W]

    return final, top_candidates, top_energies


def inference(score_model, energy_model, trend_extractor, T0, T2, device='cuda', num_candidates=10):
    """
    推理函数:生成候选T1图像并通过Energy Model筛选
    """
    # 确保 T0 和 T2 具有通道维度
    if T0.ndim == 4:  # [B, D, H, W]
        T0 = T0.unsqueeze(1)  # [B, 1, D, H, W]
    if T2.ndim == 4:  # [B, D, H, W]
        T2 = T2.unsqueeze(1)  # [B, 1, D, H, W]

    # 将数据移动到指定设备
    T0 = T0.to(device)
    T2 = T2.to(device)

    # 提取条件特征
    cond_emb = trend_extractor(T0, T2)  # [B, 64]

    # 生成候选样本
    candidates = []
    shape = T0.shape  # [B, 1, D, H, W]
    for _ in range(num_candidates):
        sample = probability_flow_sampling(
            score_model, 
            cond_emb, 
            sigma_t, 
            d_sigma_sq_dt, 
            sigma_max=50.0, 
            eps=1e-5, 
            shape=shape, 
            device=device
        )  # [B, 1, D, H, W]
        candidates.append(sample)
    candidates = torch.stack(candidates, dim=0)  # [N, B, 1, D, H, W]

    # 如果 B=1, squeeze掉批量维度
    candidates = candidates.squeeze(1)  # [N, 1, D, H, W]

    # 使用 Energy Model 进行候选筛选
    final, top_candidates, top_scores = rank_and_filter_candidates(energy_model, cond_emb, candidates, prune_ratio=0.2)

    return final, top_candidates, top_scores
