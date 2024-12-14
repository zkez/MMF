import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

import sys
sys.path.append('../')
from model.generate import ScoreModel3D, EnergyModel3D
from model.trend import TrendExtractor3D
from dataloader.dataset import LungNoduleDataset
from train_energy_model import train_energy_model
from train_score_model import train_score_model
from model.inference import inference


def main():
    # 数据加载
    csv_path = '/home/zk/MICCAI/newmainroi.csv'  
    data_dir = '/home/zk/MICCAI/roiresize'
    # text_csv_path is not needed
    # text_data = pd.read_csv(text_csv_path)
    # subject_ids = csv_data['Subject ID'].unique()
    
    csv_data = pd.read_csv(csv_path)
    subject_ids = csv_data['Subject ID'].unique()
    
    num_timesteps = 1000
    learning_rate = 0.0001
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ids, val_ids = train_test_split(subject_ids, test_size=0.2, random_state=42)
    train_data = csv_data[csv_data['Subject ID'].isin(train_ids)]
    val_data = csv_data[csv_data['Subject ID'].isin(val_ids)]
    
    train_dataset = LungNoduleDataset(train_data, data_dir, normalize=True)
    val_dataset = LungNoduleDataset(val_data, data_dir, normalize=True)
    
    # 初始化模型
    score_model = ScoreModel3D(in_channels=1, cond_dim=64, time_dim=64, base_channels=32).to(device)
    energy_model = EnergyModel3D(in_channels=1, cond_dim=64, time_dim=64, base_channels=32).to(device)
    trend_extractor = TrendExtractor3D(in_channels=2, embed_dim=64).to(device)
    
    # 1. 训练ScoreModel
    train_score_model(score_model, trend_extractor, train_dataset, epochs=10, batch_size=batch_size, lr=learning_rate, device=device)
    
    # 2. 训练EnergyModel
    train_energy_model(energy_model, trend_extractor, train_dataset, epochs=10, batch_size=batch_size, lr=learning_rate, device=device)
    
    # 3. 推理测试
    score_model.eval()
    energy_model.eval()
    trend_extractor.eval()
    
    with torch.no_grad():
        for i, (T0, T1, T2, label) in enumerate(val_dataset):
            # T0, T2: [B=1, C=1, D, H, W]
            T0 = T0.to(device)
            T1 = T1.to(device)
            T2 = T2.to(device)
            # 推理阶段不需要真实T1和label来生成，只需要T0,T2
            
            # 使用inference函数生成并筛选T1
            final, top_candidates, top_scores = inference(score_model, energy_model, trend_extractor, T0, T2, device=device, num_candidates=10)

            print("Final T1 shape:", final.shape)  # [1,1,D,H,W]
            print("Top candidates shape:", top_candidates.shape)  # [keep_num,1,D,H,W]
            print("Top scores:", top_scores)  # [keep_num]

            # 转换生成的 T1 和真实 T1 为 NumPy 格式
            generated_T1 = final.squeeze(0).squeeze(0).cpu().numpy()  # [D,H,W]
            real_T1 = T1.squeeze(0).squeeze(0).cpu().numpy()  # [D,H,W]

            # 计算相关性
            generated_T1_flat = generated_T1.flatten()  # 展平到 1D
            real_T1_flat = real_T1.flatten()  # 展平到 1D
            
            # 计算皮尔逊相关系数
            pearson_corr, _ = pearsonr(generated_T1_flat, real_T1_flat)

            print(f"Pearson Correlation between generated T1 and real T1: {pearson_corr:.4f}")

            # 如果需要保存生成结果
            # final_np = final.cpu().squeeze(0).numpy()  # [C,D,H,W]
            # np.save('/path/to/save/final_T1.npy', final_np)

            break  # 这里只处理第一个样本，移除此行可处理所有样本

if __name__ == "__main__":
    main()