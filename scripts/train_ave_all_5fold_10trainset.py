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
import io
from contextlib import redirect_stdout
from tabulate import tabulate
import tqdm
import time
import psutil
import GPUtil
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# Read configuration
config = configparser.ConfigParser()
config.read('C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\config\config.ini')

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.dataset import SealDataset
from src.model import (CNN, vgg16, vgg19, resnet50, inception_v3, resnet101,
                       resnet152, efficientnet, vit, MobileNetV2, MobileNetV3,
                       SqueezeNet, ShuffleNet, vit_tiny)
from src.transform_fig import get_transform

# Define models to train
models_to_train = ['MobileNetV2']

# Model dictionary
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

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_memory_info():
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return f"GPU Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)"
    except:
        return "GPU info not available"


def get_system_info():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    return f"CPU: {cpu_percent}% | RAM: {memory.percent}%"


def get_multiple_datasets(model_name, config, transform):
    # 获取所有训练集路径
    train_paths = []
    for key in config['data_path']:
        if key.startswith('train_path'):
            path = str(config['data_path'][key])
            train_paths.append(path)
            print(f"Found training path: {key} = {path}")

    print(f"\nTotal number of training paths found: {len(train_paths)}")

    # 创建多个训练集
    train_datasets = []
    reference_label_mapping = None

    for i, path in enumerate(train_paths):
        print(f"\nProcessing dataset {i + 1}/{len(train_paths)}")
        print(f"Path: {path}")

        # 验证路径是否存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training path does not exist: {path}")

        dataset = SealDataset(root_dir=path, transform=transform)

        # 保存第一个数据集的标签映射作为参考
        if i == 0:
            reference_label_mapping = dataset.label_to_idx
            print(f"Reference label mapping from first dataset:")
            print(reference_label_mapping)
        else:
            # 确保所有数据集使用相同的标签映射
            dataset.label_to_idx = reference_label_mapping

        print(f"Dataset {i + 1} size: {len(dataset)}")
        print(f"Dataset {i + 1} classes: {len(dataset.label_to_idx)}")

        train_datasets.append(dataset)

    # 处理验证集和测试集
    val_path = str(config['data_path']['val_path'])
    test_path = str(config['data_path']['test_path'])

    print(f"\nValidation path: {val_path}")
    print(f"Test path: {test_path}")

    # 验证路径是否存在
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation path does not exist: {val_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test path does not exist: {test_path}")

    val_dataset = SealDataset(root_dir=val_path, transform=transform)
    test_dataset = SealDataset(root_dir=test_path, transform=transform)

    # 确保验证集和测试集使用相同的标签映射
    val_dataset.label_to_idx = reference_label_mapping
    test_dataset.label_to_idx = reference_label_mapping

    print(f"\nValidation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_datasets, val_dataset, test_dataset


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
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

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=1
    )
    cm = confusion_matrix(all_labels, all_predictions)

    return total_loss / len(data_loader), precision, recall, f1, cm, all_labels, all_probs, accuracy


def run_single_experiment(model_name, trial_num, model_dict, train_dataset, val_dataset, test_dataset, device, config,
                          dataset_idx=None):
    dataset_str = f" Dataset {dataset_idx}" if dataset_idx is not None else ""
    print(f"\n{'=' * 50}")
    print(f"Starting training for {model_name}{dataset_str} - Trial {trial_num}/5")
    print(f"{'=' * 50}")

    writer = SummaryWriter(f'runs/{model_name}_dataset{dataset_idx}_trial{trial_num}_{time.strftime("%Y%m%d-%H%M%S")}')

    # Initialize model
    num_classes = len(train_dataset.label_to_idx)
    model_class = model_dict[model_name]
    model = model_class(num_classes=num_classes)
    model.to(device)

    # Initialize training components
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['train']['lr']),
        weight_decay=float(config['train']['weight_decay'])
    )
    scaler = GradScaler()

    # Create data loaders
    batch_size = int(config['train']['batch_size'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    # Training parameters
    num_epochs = int(config['train']['num_epochs'])
    patience = int(config['early_stop']['patience'])
    best_val_f1 = 0
    best_model = None
    counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'gpu_mem': get_gpu_memory_info(),
            })

        # Validation
        val_loss, val_precision, val_recall, val_f1, _, _, _, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

    # Load best model and evaluate
    if best_model is not None:
        model.load_state_dict(best_model)

    test_loss, test_precision, test_recall, test_f1, _, _, _, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )

    writer.close()

    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }


def print_model_results(model_name, results, dataset_idx=None):
    dataset_str = f" - Dataset {dataset_idx}" if dataset_idx is not None else ""
    print(f"\n{'=' * 50}")
    print(f"Results for {model_name}{dataset_str}")
    print(f"{'=' * 50}")

    headers = ["Metric", "Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5", "Mean ± Std"]
    table_data = []

    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        values = results[metric]
        mean = np.mean(values)
        std = np.std(values)
        row = [
            metric.capitalize(),
            *[f"{v:.4f}" for v in values],
            f"{mean:.4f} ± {std:.4f}"
        ]
        table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Save results
    output_dir = os.path.join(r"C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\model\output",
                              model_name)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{model_name}_dataset{dataset_idx}_results.txt")

    with open(results_file, 'w') as f:
        f.write(f"Detailed Results for {model_name}{dataset_str}\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
        f.write("\n\nRaw Values:\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = results[metric]
            mean = np.mean(values)
            std = np.std(values)
            f.write(f"\n{metric.capitalize()}:\n")
            f.write(f"Individual trials: {', '.join(f'{v:.4f}' for v in values)}\n")
            f.write(f"Mean ± Std: {mean:.4f} ± {std:.4f}\n")


def save_comprehensive_results(model_name, all_results):
    output_dir = os.path.join(r"C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\model\output",
                              model_name)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{model_name}_comprehensive_results.txt")

    with open(results_file, 'w') as f:
        f.write(f"Comprehensive Results for {model_name}\n\n")

        for dataset_name, results in all_results.items():
            f.write(f"\n{'=' * 50}\n")
            f.write(f"Results for {dataset_name}\n")
            f.write(f"{'=' * 50}\n")

            headers = ["Metric", "Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5", "Mean ± Std"]
            table_data = []

            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                values = results[metric]
                mean = np.mean(values)
                std = np.std(values)
                row = [
                    metric.capitalize(),
                    *[f"{v:.4f}" for v in values],
                    f"{mean:.4f} ± {std:.4f}"
                ]
                table_data.append(row)

            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n")


# Main execution
for model_name in models_to_train:
    print(f"\nStarting experiments for {model_name}")

    # Get all datasets
    transform = get_transform(model_name)
    train_datasets, val_dataset, test_dataset = get_multiple_datasets(model_name, config, transform)

    all_results = {}

    # Run experiments for each dataset
    for dataset_idx, train_dataset in enumerate(train_datasets, 1):
        dataset_results = defaultdict(list)

        for trial in range(5):
            results = run_single_experiment(
                model_name, trial + 1, model_dict,
                train_dataset, val_dataset, test_dataset,
                device, config, dataset_idx
            )

            for metric, value in results.items():
                dataset_results[metric].append(value)

        all_results[f"Dataset_{dataset_idx}"] = dataset_results
        print_model_results(model_name, dataset_results, dataset_idx)

    # Save comprehensive results
    save_comprehensive_results(model_name, all_results)

print("\nAll experiments completed!")