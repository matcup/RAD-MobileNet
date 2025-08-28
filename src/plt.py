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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_improved_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(10, 8))

    # 绘制微平均和宏平均ROC曲线
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Micro and Macro Average)')
    plt.legend(loc="lower right")
    plt.show()

    # 创建AUC值的热力图
    auc_values = {i: roc_auc[i] for i in range(len(roc_auc) - 2)}  # 排除micro和macro
    auc_df = pd.DataFrame.from_dict(auc_values, orient='index', columns=['AUC'])
    auc_df = auc_df.sort_values('AUC', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.heatmap(auc_df.T, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('AUC Values for Each Class')
    plt.show()

    # 打印AUC值低于某个阈值的类别
    threshold = 0.95
    poor_performers = auc_df[auc_df['AUC'] < threshold]
    print(f"Classes with AUC < {threshold}:")
    print(poor_performers)



def calculate_roc_auc(y_true, y_score):
    n_classes = y_score.shape[1]

    # Binarize the output
    y_test = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)

    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(fpr) - 2))
    for i, color in zip(range(len(fpr) - 2), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()



