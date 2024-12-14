import os
import torch
import time
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/home/wangchangmiao/fzj/zk/MFF')
from dataloader.dataset import  LungNoduleTextDataset
from utils.observer import Runtime_Observer


def train_model(model, train_loader, val_loader, device, optimizer, BCE_criterion, MSE_criterion, num_epochs, scheduler=None):
    train_losses = []
    val_losses = []
    auc_scores = []
    accuracies = [] 
    epoch_steps = len(train_loader)
    best_val_loss = float('inf')
    start_time = time.time()
    observer.log("start training\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        observer.reset()

        # 训练阶段
        with tqdm(total=epoch_steps, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for batch in train_loader:
                inputs1, inputs2, labels, text = batch['T0_image'], batch['T1_image'], batch['label'], batch['table_info']
                inputs1 = inputs1.unsqueeze(1).to(device)  
                inputs2 = inputs2.unsqueeze(1).to(device)  
                text = torch.squeeze(text).to(device).to(torch.float32)

                labels = labels.to(device, dtype=torch.long)
                label_onehot = F.one_hot(labels, num_classes=2).to(torch.float32).to(device)

                feats, yhat = model(inputs1, inputs2, text, concat_type='train')
                loss = BCE_criterion(yhat, label_onehot)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                aux_scheduler.step()

                if scheduler:
                    scheduler.step()
                running_loss += loss.item()*inputs1.size(0)

                _, preds = torch.max(yhat, 1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

                pbar.set_postfix({'Loss': running_loss / (pbar.n + 1),'Accuracy': correct_predictions / total_samples})
                pbar.update()

        train_losses = running_loss / (len(train_loader.dataset))
        accuracies.append(correct_predictions / total_samples)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        correct_predictions = 0
        total_samples = 0
        observer.log(f"Loss: {train_losses:.4f}\n")

        with torch.no_grad():
            for batch in val_loader:
                inputs1, inputs2, labels, _ = batch['T0_image'], batch[
                    'T1_image'], batch['label'], batch['table_info']
                inputs1 = inputs1.unsqueeze(1).to(device)  
                inputs2 = inputs2.unsqueeze(1).to(device)
                text = torch.squeeze(text).to(device).to(torch.float32)

                labels = labels.to(device, dtype=torch.long)

                feats, yhat = model(inputs1, inputs2, text, concat_type='train')
                loss = BCE_criterion(yhat, F.one_hot(labels, num_classes=2).to(torch.float32).to(device))
                val_loss += loss.item()*inputs1.size(0)

                probabilities = torch.softmax(yhat, dim=1)
                _, predictions = torch.max(yhat, dim=1)
                confidence_scores = probabilities[range(len(predictions)), 1]
                observer.update(predictions, labels, confidence_scores)

            val_losses = val_loss / len(val_loader)
            observer.log(f"Test Loss: {val_losses:.4f}\n")

        observer.record_loss(epoch, train_losses, val_losses)
        if observer.excute(epoch):
            print("Early stopping")
            break

        # 如果是最优模型，保存权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
        final_model = model
    torch.save(best_model.state_dict(), 'outputs/model_best_Hifuse_small.pth')
    torch.save(final_model.state_dict(), 'outputs/model_final_Hifuse_small.pth')
    
    end_time = time.time()
    observer.log(f"\nRunning time: {end_time - start_time:.2f} second\n")
    observer.finish()
    
    return train_losses, val_losses, auc_scores, accuracies


if __name__ == '__main__':
    # 数据加载
    csv_path = '/home/wangchangmiao/yuxiao/CSF-NET-main/CSF-NET-main/datasets/newmainroi.csv'  
    data_dir = '/home/wangchangmiao/sy/isbi/roiresize/'
    seg_dir = '/home/shenyinhd/cnnlstm/seg/'
    text_csv_path = '/home/wangchangmiao/yuxiao/CSF-NET-main/CSF-NET-main/datasets/scale_information.csv'
    csv_data = pd.read_csv(csv_path)
    text_data = pd.read_csv(text_csv_path)
    subject_ids = csv_data['Subject ID'].unique()
    
    # 模型训练
    num_timesteps = 1000
    learning_rate = 0.0001
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ids, val_ids = train_test_split(subject_ids, test_size=0.2, random_state=42)
    train_data = csv_data[csv_data['Subject ID'].isin(train_ids)]
    val_data = csv_data[csv_data['Subject ID'].isin(val_ids)]
    
    train_dataset = LungNoduleTextDataset(train_data, data_dir, text_data, normalize=True, augment_minority_class=False)
    val_dataset = LungNoduleTextDataset(val_data, data_dir, text_data, normalize=True)
    
    # 定义模型
    model = BuildModel().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    aux_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_mult=2, T_0=80, eta_min=1e-4)
    
    # 创建Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化记录器
    best_val_loss = float('inf') 
    train_losses, val_losses = [], []
    val_accuracies, val_aucs, val_f1_scores = [], [], []
    
    BCE_criterion = nn.BCELoss()
    MSE_criterion = nn.MSELoss()
    
    if not os.path.exists(f"debug"):
        os.makedirs(f"debug")
    observer = Runtime_Observer(log_dir=f"debug", device=device, name="debug", seed=42)
    
    model.to(device)
    
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()

    print("\n===============================================\n")
    print("model parameters: " + str(num_params))
    print("\n===============================================\n")
    
    train_losses, val_losses, auc_scores, _ = train_model(model, train_loader, val_loader, device, optimizer, BCE_criterion, MSE_criterion, num_epochs=100)
