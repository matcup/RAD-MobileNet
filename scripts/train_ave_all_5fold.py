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
                       SqueezeNet, ShuffleNet, vit_tiny, DilatedMobileNetV2)
from src.transform_fig import get_transform

# Define the list of models to train
# models_to_train = [
#     'CNN', 'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152', 'inception_v3',
#     'efficientnet', 'vit', 'MobileNetV2', 'MobileNetV3', 'ShuffleNet', 'SqueezeNet'
# ]

models_to_train = [
    'DilatedMobileNetV2'
]

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
    'vit_tiny': vit_tiny,
    'DilatedMobileNetV2': DilatedMobileNetV2
}

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_memory_info():
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Assuming using first GPU
            return f"GPU Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)"
    except:
        return "GPU info not available"


def get_system_info():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    return f"CPU: {cpu_percent}% | RAM: {memory.percent}%"


# Function to evaluate the model
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


def run_single_experiment(model_name, trial_num, model_dict, train_dataset, val_dataset, test_dataset, device, config):
    print(f"\n{'=' * 50}")
    print(f"Starting training for {model_name} - Trial {trial_num}/5")
    print(f"{'=' * 50}")

    writer = SummaryWriter(f'runs/{model_name}_trial{trial_num}_{time.strftime("%Y%m%d-%H%M%S")}')

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
        val_loss, val_precision, val_recall, val_f1, _, _, _, val_accuracy = evaluate(model, val_loader, criterion,
                                                                                      device)

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


def print_model_results(model_name, results):
    print(f"\n{'=' * 50}")
    print(f"Results for {model_name}")
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

    # Save individual model results to file
    output_dir = os.path.join(r"C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\model\output",
                              model_name)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{model_name}_results.txt")

    with open(results_file, 'w') as f:
        f.write(f"Detailed Results for {model_name}\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
        f.write("\n\nRaw Values:\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = results[metric]
            mean = np.mean(values)
            std = np.std(values)
            f.write(f"\n{metric.capitalize()}:\n")
            f.write(f"Individual trials: {', '.join(f'{v:.4f}' for v in values)}\n")
            f.write(f"Mean ± Std: {mean:.4f} ± {std:.4f}\n")

    print(f"\nDetailed results saved to {results_file}")


# Dictionary to store results for all trials
model_results = defaultdict(lambda: defaultdict(list))

# Run experiments for each model
for model_name in models_to_train:
    print(f"\nStarting experiments for {model_name}")

    # Create datasets once for each model
    transform = get_transform(model_name)
    train_dataset = SealDataset(root_dir=str(config['data_path']['train_path']), transform=transform)
    val_dataset = SealDataset(root_dir=str(config['data_path']['val_path']), transform=transform)
    test_dataset = SealDataset(root_dir=str(config['data_path']['test_path']), transform=transform)
    print(str(config['data_path']['train_path']))
    # Run 5 trials
    for trial in range(5):
        results = run_single_experiment(
            model_name, trial + 1, model_dict,
            train_dataset, val_dataset, test_dataset,
            device, config
        )

        # Store results
        for metric, value in results.items():
            model_results[model_name][metric].append(value)

        # Print results after each trial
        print(f"\nTrial {trial + 1} Results for {model_name}:")
        for metric, value in results.items():
            print(f"{metric.capitalize()}: {value:.4f}")

    # Print and save comprehensive results for this model
    print_model_results(model_name, model_results[model_name])

# Print final summary of all models
print("\nFinal Summary of All Models:")
headers = ["Model", "Accuracy", "Precision", "Recall", "F1"]
summary_data = []

for model_name in models_to_train:
    row = [model_name]
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        values = model_results[model_name][metric]
        mean = np.mean(values)
        std = np.std(values)
        row.append(f"{mean:.4f} ± {std:.4f}")
    summary_data.append(row)

print(tabulate(summary_data, headers=headers, tablefmt="grid"))

# Save final summary
summary_file = os.path.join(r"C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\model\output",
                            "final_summary.txt")
with open(summary_file, 'w') as f:
    f.write("Final Summary of All Models\n\n")
    f.write(tabulate(summary_data, headers=headers, tablefmt="grid"))

print(f"\nFinal summary saved to {summary_file}")