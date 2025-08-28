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

config = configparser.ConfigParser()
config.read('C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\config\config.ini')

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.dataset import SealDataset
from src.model import CNN, vgg16, vgg19, resnet50, inception_v3, resnet101, resnet152, efficientnet, vit, MobileNetV2, \
    MobileNetV3, SqueezeNet, ShuffleNet, vit_tiny
from src.transform_fig import get_transform

# Define the list of models to train
models_to_train = [
    'CNN', 'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152', 'inception_v3',
    'efficientnet', 'vit', 'MobileNetV2', 'MobileNetV3', 'ShuffleNet', 'SqueezeNet'
]

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
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    correct = 0  # 添加正确预测计数
    total = 0  # 添加总样本计数
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

            # 计算accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total  # 计算Overall Accuracy
    cm = confusion_matrix(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro',
                                                               zero_division=1)

    return total_loss / len(data_loader), precision, recall, f1, cm, all_labels, all_probs, accuracy


# Function to save plot
def save_plot(fig, model_name, plot_type):
    if fig is None:
        print(f"Error: Figure object is None for {model_name} {plot_type}")
        return
    output_dir = r"C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\model\plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_{plot_type}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)


def calculate_roc_auc(y_test, y_score):
    try:
        n_classes = y_score.shape[1]
        y_test_bin = label_binarize(y_test, classes=range(n_classes))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return fpr, tpr, roc_auc
    except Exception as e:
        print(f"Error in calculate_roc_auc: {e}")
        return None, None, None


def plot_improved_roc_curve(fpr, tpr, roc_auc):
    if fpr is None or tpr is None or roc_auc is None:
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)

        ax.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
                color='navy', linestyle=':', linewidth=4)

        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        plt.close(fig)  # Close the figure to prevent display
        return fig
    except Exception as e:
        print(f"Error in plot_improved_roc_curve: {e}")
        return None


# Create a dictionary to store model performance
model_performance = {}

# Main training loop
for model_name in models_to_train:
    print(f"\n{'=' * 50}")
    print(f"Starting training for {model_name}")
    print(f"{'=' * 50}")

    # Initialize tensorboard writer
    writer = SummaryWriter(f'runs/{model_name}_{time.strftime("%Y%m%d-%H%M%S")}')

    # Redirect stdout to a string buffer
    f = io.StringIO()
    with redirect_stdout(f):
        # Get data transform
        transform = get_transform(model_name)

        # Create datasets
        train_path = config['data_path']['train_path']
        val_path = config['data_path']['val_path']
        test_path = config['data_path']['test_path']

        train_dataset = SealDataset(root_dir=str(train_path), transform=transform)
        val_dataset = SealDataset(root_dir=str(val_path), transform=transform)
        test_dataset = SealDataset(root_dir=str(test_path), transform=transform)

        # Create index to label mapping
        idx_to_label = {v: k for k, v in train_dataset.label_to_idx.items()}

        # Print dataset information
        print(f"Training samples: {len(train_dataset)}", f"Validation samples: {len(val_dataset)}",
              f"Test samples: {len(test_dataset)}", f"Training classes: {len(train_dataset.label_to_idx)}",
              f"Test classes: {len(test_dataset.label_to_idx)}")

        # Create data loaders
        batch_size = int(config['train']['batch_size'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

        # Initialize model
        num_classes = len(train_dataset.label_to_idx)
        model_class = model_dict[model_name]
        model = model_class(num_classes=num_classes)

        # Print model parameters
        print("\nModel Parameters:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        lr = float(config['train']['lr'])
        weight_decay = float(config['train']['weight_decay'])
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Print training parameters
        print("\nTraining Parameters:")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {weight_decay}")
        print(f"Batch size: {batch_size}")
        print(f"Number of epochs: {config['train']['num_epochs']}")
        print(f"Early stopping patience: {config['early_stop']['patience']}")

        # Set device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        model.to(device)
        criterion.to(device)

        # Initialize GradScaler for mixed precision training
        scaler = GradScaler()

        # Training loop
        num_epochs = int(config['train']['num_epochs'])
        patience = int(config['early_stop']['patience'])
        best_val_f1 = 0
        best_model = None
        counter = 0
        total_training_time = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            running_loss = 0.0

            # Initialize progress bar for batches
            pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            batch_count = 0

            for images, labels in pbar:
                batch_start_time = time.time()
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                batch_count += 1

                # Update progress bar
                batch_time = time.time() - batch_start_time
                current_lr = optimizer.param_groups[0]['lr']

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{running_loss / batch_count:.4f}',
                    'batch_time': f'{batch_time:.2f}s',
                    'lr': f'{current_lr:.2e}',
                    'gpu_mem': get_gpu_memory_info(),
                })

                # Log to tensorboard
                global_step = epoch * len(train_loader) + batch_count
                writer.add_scalar('training_loss', loss.item(), global_step)
                writer.add_scalar('learning_rate', current_lr, global_step)

            # Calculate metrics
            train_loss, train_precision, train_recall, train_f1, _, _, _, train_accuracy = evaluate(model, train_loader)
            val_loss, val_precision, val_recall, val_f1, _, _, _, val_accuracy = evaluate(model, val_loader)

            epoch_time = time.time() - epoch_start_time
            total_training_time += epoch_time

            # Print epoch summary
            print(f'\nEpoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.2f}s')
            print(f'System Info: {get_system_info()}')
            print(f'GPU Info: {get_gpu_memory_info()}')
            print(f'Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, '
                  f'Recall: {train_recall:.4f}, F1: {train_f1:.4f}')
            print(f'Val - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, '
                  f'Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

            # Log to tensorboard
            writer.add_scalars('epoch_metrics', {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_f1': train_f1,
                'val_f1': val_f1
            }, epoch)

            # Early stopping logic
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = model.state_dict()
                counter = 0
                print(f'New best model saved! (F1: {val_f1:.4f})')
            else:
                counter += 1
                print(f'Early stopping counter: {counter}/{patience}')
                if counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    break

        print(f"\nTraining completed for {model_name}")
        print(f"Total training time: {total_training_time / 3600:.2f} hours")
        print(f"Best validation F1: {best_val_f1:.4f}")

        # Close tensorboard writer
        writer.close()

        # Load best model
        if best_model is not None:
            model.load_state_dict(best_model)

        # Evaluate on test set
        test_loss, test_precision, test_recall, test_f1, cm, test_labels, test_probs, test_accuracy = evaluate(model, test_loader)
        print(f'Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, '
              f'Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

        # Save model performance
        model_performance[model_name] = {
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1': test_f1
        }

        # Calculate and plot ROC curve
        fpr, tpr, roc_auc = calculate_roc_auc(np.array(test_labels), np.array(test_probs))
        if fpr is not None and tpr is not None and roc_auc is not None:
            fig = plot_improved_roc_curve(fpr, tpr, roc_auc)
            if fig is not None:
                save_plot(fig, model_name, "ROC_curve")

            # Print micro and macro average AUC-ROC
            print(f'Micro-average AUC-ROC: {roc_auc["micro"]:.4f}')
            print(f'Macro-average AUC-ROC: {roc_auc["macro"]:.4f}')

        # Print confusion matrix if configured
        cm_ = int(config['results']['cm_show'])
        if cm_ == 0:
            print("\nConfusion Matrix:")
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)

            labels = [idx_to_label[i] for i in range(len(idx_to_label))]

            print("True\Pred", end="\t")
            for label in labels:
                print(f"{label[:7]:<7}", end="\t")
            print()

            for i, row in enumerate(cm):
                print(f"{labels[i][:7]:<7}", end="\t")
                for cell in row:
                    print(f"{cell:<7}", end="\t")
                print()

        # Save best model
        torch.save(model.state_dict(), f'best_seal_model_{model_name}.pth')

    # Save the output to a file
    output_dir = r"/Chinese-Seal-Recognize/model/result/output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_output.txt")
    with open(output_file, 'w') as file:
        file.write(f.getvalue())

    print(f"Output for {model_name} saved to {output_file}")

# Print performance table
print("\nModel Performance on Test Set:")
headers = ["Model", "Accuracy", "Precision", "Recall", "F1"]
table_data = [[model,
               f"{perf['Accuracy']:.4f}",
               f"{perf['Precision']:.4f}",
               f"{perf['Recall']:.4f}",
               f"{perf['F1']:.4f}"]
              for model, perf in model_performance.items()]
print(tabulate(table_data, headers=headers, tablefmt="grid"))

print("Training completed for all models.")