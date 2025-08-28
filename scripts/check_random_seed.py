import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import configparser
import numpy as np
from torch.amp import autocast, GradScaler
import platform
import time
import random
import psutil
import GPUtil
from datetime import datetime


def set_seed(seed=42):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_environment():
    """检查运行环境并打印相关信息"""
    print("\n=== 环境检查 ===")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"cuDNN版本: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"CPU核心数: {psutil.cpu_count()}")
    print(f"物理内存: {psutil.virtual_memory().total / (1024.0 ** 3):.1f} GB")


def monitor_system_resources():
    """监控系统资源使用情况"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()

    print("\n=== 系统资源监控 ===")
    print(f"CPU使用率: {cpu_percent}%")
    print(f"内存使用: {memory.used / (1024.0 ** 3):.1f}GB / {memory.total / (1024.0 ** 3):.1f}GB ({memory.percent}%)")

    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        print(f"GPU内存使用: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)")
        print(f"GPU负载: {gpu.load * 100:.1f}%")


def verify_data_loading(train_loader, train_dataset, val_dataset, test_dataset):
    """验证数据加载的正确性和统计信息"""
    print("\n=== 数据加载验证 ===")
    first_batch = next(iter(train_loader))
    images, labels = first_batch

    print(f"Batch 图像形状: {images.shape}")
    print(f"Batch 标签形状: {labels.shape}")
    print(f"图像数值范围: [{images.min():.3f}, {images.max():.3f}]")
    print(f"图像均值: {images.mean():.3f}")
    print(f"图像标准差: {images.std():.3f}")

    print("\n数据集大小:")
    print(f"训练集: {len(train_dataset)}")
    print(f"验证集: {len(val_dataset)}")
    print(f"测试集: {len(test_dataset)}")


def check_model_state(model):
    """检查模型状态和参数统计信息"""
    print("\n=== 模型状态检查 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型结构:\n{model}")


def evaluate(model, data_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算评估指标
    cm = confusion_matrix(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions,
                                                               average='macro', zero_division=1)

    return total_loss / len(data_loader), precision, recall, f1, cm


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    batch_times = []
    correct = 0
    total = 0

    epoch_start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if batch_idx % 10 == 0:
            print(f'\nBatch [{batch_idx}/{len(train_loader)}]')
            print(f'Loss: {loss.item():.4f}')
            print(f'Accuracy: {100.0 * correct / total:.2f}%')
            print(f'Batch Time: {batch_time:.3f}s')
            monitor_system_resources()

    epoch_time = time.time() - epoch_start_time
    return running_loss / len(train_loader), 100.0 * correct / total, np.mean(batch_times), epoch_time


def save_checkpoint(state, filename):
    """保存检查点"""
    print(f"\n保存检查点到 {filename}")
    torch.save(state, filename)


def main():
    # 设置随机种子
    set_seed(42)

    # 检查环境
    check_environment()

    # 读取配置文件
    config = configparser.ConfigParser()
    config.read('C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\config\config.ini')

    # 设置路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

    from src.dataset import SealDataset
    from src.model import CNN, vgg16, vgg19, resnet50, inception_v3, resnet101, resnet152, efficientnet, vit, \
        MobileNetV2, \
        MobileNetV3, SqueezeNet, ShuffleNet, vit_tiny
    from src.transform_fig import get_transform

    # 模型字典
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

    # 获取模型配置
    model_name = config['models']['name']
    print(f'\n使用模型: {model_name}')
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found. Available models: {', '.join(model_dict.keys())}")

    # 准备数据
    transform = get_transform(model_name)
    feature_ = int(config['train']['feature'])

    train_dataset = SealDataset(root_dir=config['data_path']['train_path'], transform=transform, feature_=feature_)
    val_dataset = SealDataset(root_dir=config['data_path']['val_path'], transform=transform, feature_=feature_)
    test_dataset = SealDataset(root_dir=config['data_path']['test_path'], transform=transform, feature_=feature_)

    batch_size = int(config['train']['batch_size'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    # 验证数据加载
    verify_data_loading(train_loader, train_dataset, val_dataset, test_dataset)

    # 初始化模型
    num_classes = len(train_dataset.label_to_idx)
    model = model_dict[model_name](num_classes=num_classes)

    # 检查模型状态
    check_model_state(model)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 初始化训练组件
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=float(config['train']['lr']),
                           weight_decay=float(config['train']['weight_decay']))
    scaler = GradScaler()

    # 训练循环
    num_epochs = int(config['train']['num_epochs'])
    patience = int(config['early_stop']['patience'])
    best_val_f1 = 0
    best_model = None
    counter = 0

    # 创建日志文件
    log_dir = 'training_logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    print(f"\n开始训练...")

    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')

        # 训练一个epoch
        train_loss, train_acc, avg_batch_time, epoch_time = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch)

        # 验证
        val_loss, val_precision, val_recall, val_f1, _ = evaluate(model, val_loader, criterion, device)

        # 记录训练信息
        log_info = (f"Epoch {epoch + 1}/{num_epochs}\n"
                    f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%\n"
                    f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}\n"
                    f"Epoch Time: {epoch_time:.2f}s, Avg Batch Time: {avg_batch_time:.3f}s\n")

        with open(log_file, 'a') as f:
            f.write(log_info + '\n')

        print(log_info)

        # 早停检查
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict()
            counter = 0

            # 保存最佳模型
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, f'best_model_{model_name}.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

    # 加载最佳模型进行最终评估
    if best_model is not None:
        model.load_state_dict(best_model)

    # 最终测试
    test_loss, test_precision, test_recall, test_f1, cm = evaluate(model, test_loader, criterion, device)

    print("\n=== 最终测试结果 ===")
    print(f'Loss: {test_loss:.4f}')
    print(f'Precision: {test_precision:.4f}')
    print(f'Recall: {test_recall:.4f}')
    print(f'F1 Score: {test_f1:.4f}')

    # 保存最终结果
    results_info = (f"\nFinal Results:\n"
                    f"Test Loss: {test_loss:.4f}\n"
                    f"Test Precision: {test_precision:.4f}\n"
                    f"Test Recall: {test_recall:.4f}\n"
                    f"Test F1: {test_f1:.4f}\n"
                    f"Confusion Matrix:\n{cm}\n")

    with open(log_file, 'a') as f:
        f.write(results_info)


if __name__ == "__main__":
    main()