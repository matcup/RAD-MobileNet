import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import configparser
import numpy as np
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\config\config.ini')

# 将 src 目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到Python路径
sys.path.append(project_root)

from src.dataset import SealDataset
from src.model import CNN, vgg16, vgg19, resnet50, inception_v3, resnet101, resnet152, efficientnet, vit, MobileNetV2, \
    MobileNetV3, SqueezeNet, ShuffleNet, vit_tiny
from src.transform_fig import get_transform
from src.plt import *

# 获取模型名称
model_dict = {
    'vgg16': vgg16,
    'vgg19': vgg19,
    'inception_v3': inception_v3,
    'CNN': CNN,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'efficientnet': efficientnet,
    'vit': vit,
    'MobileNetV2': MobileNetV2,
    'MobileNetV3': MobileNetV3,
    'SqueezeNet': SqueezeNet,
    'ShuffleNet': ShuffleNet,
    'vit_tiny': vit_tiny
}
model_name = config['models']['name']
print('当前使用的模型为：', model_name)
print('相关配置为:', config['train']['batch_size'], config['train']['num_epochs'], config['train']['lr'],
      config['train']['feature'], config['train']['weight_decay'])
if model_name not in model_dict:
    raise ValueError(f"Model {model_name} not found. Available models are: {', '.join(model_dict.keys())}")

# 获取数据转换
transform = get_transform(model_name)

# 创建训练、验证和测试数据集
train_path = config['data_path']['train_path']
val_path = config['data_path']['val_path']
test_path = config['data_path']['test_path']

train_dataset = SealDataset(root_dir=str(train_path), transform=transform)
val_dataset = SealDataset(root_dir=str(val_path), transform=transform)
test_dataset = SealDataset(root_dir=str(test_path), transform=transform)

# 创建索引到标签的映射
idx_to_label = {v: k for k, v in train_dataset.label_to_idx.items()}

# 打印数据集信息
print(f"训练集样本数量: {len(train_dataset)}", f"验证集样本数量: {len(val_dataset)}", f"验证集样本数量: {len(val_dataset)}", \
      f"测试集样本数量: {len(test_dataset)}", f"训练集类别数量: {len(train_dataset.label_to_idx)}", f"测试集类别数量: {len(test_dataset.label_to_idx)}")

# 创建数据加载器
batch_size = int(config['train']['batch_size'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

# 初始化模型
num_classes = len(train_dataset.label_to_idx)
model_class = model_dict[model_name]
model = model_class(num_classes=num_classes)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=float(config['train']['lr']),
                       weight_decay=float(config['train']['weight_decay']))

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第n个 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
criterion.to(device)

# 初始化 GradScaler 用于混合精度训练
scaler = GradScaler()


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    cm = confusion_matrix(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro',
                                                               zero_division=1)
    return total_loss / len(data_loader), precision, recall, f1, cm, all_labels, all_probs





# 训练循环
num_epochs = int(config['train']['num_epochs'])
patience = int(config['early_stop']['patience'])
best_val_f1 = 0
best_model = None
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # 使用 autocast 进行混合精度计算
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 使用 scaler 来缩放损失，执行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    # 计算训练集的指标
    train_loss, train_precision, train_recall, train_f1, _, _, _ = evaluate(model, train_loader)

    # 计算验证集的指标
    val_loss, val_precision, val_recall, val_f1, _, _, _ = evaluate(model, val_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    print(
        f'Train - Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')
    print(f'Val - Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

    # 早停逻辑
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

# 可选：保存检查点
if 'model_path' in config['train']:
    model_path = config['train']['model_path']
    os.makedirs(model_path, exist_ok=True)

    checkpoint_filename = f'checkpoint_epoch_{epoch + 1}.pth'
    checkpoint_path = os.path.join(model_path, checkpoint_filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_f1': val_f1,
    }, checkpoint_path)
else:
    print("Warning: 'model_path' not specified in config. Skipping checkpoint saving.")

# 加载最佳模型
if best_model is not None:
    model.load_state_dict(best_model)

# 在测试集上评估模型
test_loss, test_precision, test_recall, test_f1, cm, test_labels, test_probs = evaluate(model, test_loader)
print(f'Test - Loss: {test_loss:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

# 计算并绘制ROC曲线
fpr, tpr, roc_auc = calculate_roc_auc(np.array(test_labels), np.array(test_probs))
plot_improved_roc_curve(fpr, tpr, roc_auc)

# 打印微平均和宏平均AUC-ROC
print(f'Micro-average AUC-ROC: {roc_auc["micro"]:.4f}')
print(f'Macro-average AUC-ROC: {roc_auc["macro"]:.4f}')

cm_ = int(config['results']['cm_show'])
if cm_ == 0:
    print("\n混淆矩阵:")
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # 创建标签列表
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]

    # 打印列标签
    print("True\Pred", end="\t")
    for label in labels:
        print(f"{label[:7]:<7}", end="\t")  # 截断标签名称以适应屏幕宽度
    print()

    # 打印混淆矩阵和行标签
    for i, row in enumerate(cm):
        print(f"{labels[i][:7]:<7}", end="\t")  # 行标签
        for cell in row:
            print(f"{cell:<7}", end="\t")
        print()
else:
    pass

# 保存最佳模型
torch.save(model.state_dict(), 'best_seal_model.pth')