import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import configparser
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
config = configparser.ConfigParser()
config.read('C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\config\config.ini')



# 1. 添加新的特征提取模块
class SealFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 形状特征提取
        self.shape_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 纹理特征提取
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 局部细节特征提取
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1)
        )

    def forward(self, x):
        shape_features = self.shape_encoder(x)
        texture_features = self.texture_encoder(x)
        detail_features = self.detail_encoder(x)
        return shape_features, texture_features, detail_features


# 2. 添加注意力机制
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(pool))
        return x * attention


# 3. 特征融合模块
class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = SpatialAttention()
        self.conv1x1 = nn.Conv2d(in_channels * 3, in_channels, 1)

    def forward(self, shape_feat, texture_feat, detail_feat):
        # 应用注意力
        shape_feat = self.attention(shape_feat)
        texture_feat = self.attention(texture_feat)
        detail_feat = self.attention(detail_feat)

        # 特征融合
        concat_features = torch.cat([shape_feat, texture_feat, detail_feat], dim=1)
        fused_features = self.conv1x1(concat_features)
        return fused_features


# 4. 改进的主模型结构
class ImprovedSealModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = SealFeatureExtractor()
        self.feature_fusion = FeatureFusion(32)

        # 级联分类器
        self.classifier1 = nn.Sequential(
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.final_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # 特征提取
        shape_features, texture_features, detail_features = self.feature_extractor(x)

        # 特征融合
        fused_features = self.feature_fusion(shape_features, texture_features, detail_features)

        # 级联分类
        features = fused_features.view(fused_features.size(0), -1)
        features = self.classifier1(features)
        features = self.classifier2(features)
        output = self.final_classifier(features)

        return output


# 5. 修改训练函数
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


# 6. 修改评估函数
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=1
    )

    return total_loss / len(data_loader), precision, recall, f1, accuracy


# 7. 主训练流程
def run_training(model_name, train_dataset, val_dataset, test_dataset, device, config):
    model = ImprovedSealModel(num_classes=len(train_dataset.label_to_idx)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['train']['lr']),
        weight_decay=float(config['train']['weight_decay'])
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config['train']['num_epochs'])
    )

    scaler = GradScaler()
    writer = SummaryWriter(f'runs/{model_name}')

    best_val_f1 = 0
    patience = int(config['early_stop']['patience'])
    counter = 0

    for epoch in range(int(config['train']['num_epochs'])):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # 验证
        val_loss, val_precision, val_recall, val_f1, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 早停
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'checkpoint_{model_name}.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

        # 记录日志
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    # 测试集评估
    model.load_state_dict(torch.load(f'checkpoint_{model_name}.pt'))
    test_loss, test_precision, test_recall, test_f1, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    return {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }