import skimage
import skimage.io as io
import skimage.transform as transform
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import metrics
import cv2
import configparser
import torch
import PIL as Image


def compute_Pn_Pa(image):
    """
    image为灰度图
    Pn为连通域数量
    Pa为像素数量
    """
    a = cv2.connectedComponentsWithStats(image)
    img_shape = image.shape
    WH = img_shape[0] * img_shape[1]
    Pn = a[0] / WH
    image = np.array(image)
    S_r = np.where(image, 0, 1)
    S_r = np.sum(S_r)
    Pa = 1 - (S_r / WH)

    return Pn, Pa


def GetRed(img, H):
    """
    提取图中的红色部分
    """
    # 转化为hsv空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv = img
    img_shape = hsv.shape
    # print(hsv.shape)

    # 颜色在HSV空间下的上下限156-180还能改成0-10
    # low_hsv = np.array([0, 43, 46])
    # high_hsv = np.array([16, 255, 255])
    low_hsv = np.array([0, 50, 50])
    high_hsv = np.array([int(H), 255, 255])

    # 使用opencv的inRange函数提取颜色
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    Red = cv2.bitwise_and(img, img, mask=mask)
    # Red = cv2.bitwise_not(Red, Red, mask=mask)
    return mask


def adopt_HSV_H(image):
    """
    自适应调整H的数值；
    :param image: 输入图片
    :return: 返回H的值
    闭运算（closing）：通常用于填充小孔或小裂缝
    开运算（opening）：通常用于去除小的明亮区域（"斑点噪声"）
    高斯滤波：用于平滑图像
    """
    H = 25
    Pn_range = [1e-5, 6e-4]
    Pa_range = [0.2, 0.42]
    image = skimage.exposure.rescale_intensity(image)

    try:
        for i in range(int(H)):
            image_ = GetRed(image, H)
            Pn, Pa = compute_Pn_Pa(image_)
            # print(Pn, Pa)
            if (Pn < Pn_range[0] or Pn > Pn_range[1]) or (Pa < Pa_range[0] or Pa > Pa_range[1]):
                H -= 1
            else:
                break
    except:
        image_ = GetRed(image, 15)
    if H == 0:
        image_ = GetRed(image, 14)
        Pn, Pa = compute_Pn_Pa(image_)
    kernel_size = 1  # 可以根据需要调整
    image_ = skimage.morphology.closing(image_)
    image_ = skimage.morphology.opening(image_, skimage.morphology.disk(kernel_size))
    image_ = skimage.filters.gaussian(image_, sigma=0.1)
    return image_, H, Pn, Pa


def Get_label(path):
    y = []
    img_class_path = os.listdir(path)[:]
    for i in img_class_path[:]:
        k = str(i).strip('.png')
        # 如果存在标记序号则去除，如果不在label列表，则加入。
        try:
            if k[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                # 去除尾部标记序号
                l = k[1][:-1]
                if l not in y:
                    y.append(l)
            else:
                l = k
                y.append(l)
        except:
            print(k)
    return list(y)


def read_file_img(path, label):
    x = []
    y = []
    img_class_path = os.listdir(path)[:]
    for i in img_class_path[:]:
        address = str(path + '\\' + str(i))
        x_ = cv2.imdecode(np.fromfile(address, dtype=np.uint8), 1)
        gray = cv2.cvtColor(x_, cv2.COLOR_BGR2GRAY)
        x_o = x_
        # x_ = skimage.filters.gaussian(x_, sigma=2)
        # x_ = skimage.feature.canny(x_)
        x_, H, Pn, Pa = adopt_HSV_H(x_)
        # x_ = GetRed(x_, H)
        x_r = x_
        if len(x_.shape) == 3:
            x_ = x_[:, :, 0]
        elif len(x_.shape) == 2:
            x_ = x_[:, :]
        x_ = skimage.filters.gaussian(x_, sigma=0.2)
        x_ = skimage.feature.canny(x_, sigma=2.5)
        x_ = skimage.filters.gaussian(x_, sigma=0.5)
        x_ = np.expand_dims(x_, axis=2)
        x_r = np.expand_dims(x_r, axis=2)
        gray = np.expand_dims(gray, axis=2)

        # x_ = np.append(x_, x_o, axis=2)
        x_ = np.append(x_, x_r, axis=2)
        x_ = np.append(x_, gray, axis=2)

        # print(x_.shape)
        x_ = transform.resize(x_, (200, 200, 3))
        k = str(i).strip('.png')
        # 如果存在标记序号则去除，如果不在label列表，则加入。
        try:
            if k[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                # 去除尾部标记序号
                l = k[:-1]
                y.append(int(label.index(str(l))))
            else:
                l = k
                y.append(int(label.index(str(l))))
        except:
            print(k)
        # print(l)
        try:
            y.append(label.index(str(l)))
            x.append(x_)
        except:
            pass
    return x, y


def adopt_HSV_H_(image, specified_H=None):
    """
    自适应调整H的数值，或使用指定的H值，并提取边缘特征和灰度特征；
    :param image: 输入图片
    :param specified_H: 指定的H值，如果为None则使用自适应方法
    :return: 返回处理后的图像、H值、Pn、Pa、边缘特征和灰度特征
    """
    Pn_range = [1e-5, 6e-3]
    Pa_range = [0.2, 0.42]
    image = skimage.exposure.rescale_intensity(image)

    # 提取灰度特征
    gray_features = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if specified_H is not None:
        # 使用指定的H值
        H = specified_H
        image_ = GetRed(image, H)
        Pn, Pa = compute_Pn_Pa(image_)
    else:
        # 使用原有的自适应方法
        H = 25
        try:
            for i in range(int(H)):
                image_ = GetRed(image, H)
                Pn, Pa = compute_Pn_Pa(image_)
                if (Pn < Pn_range[0] or Pn > Pn_range[1]) or (Pa < Pa_range[0] or Pa > Pa_range[1]):
                    H -= 1
                else:
                    break
        except:
            image_ = GetRed(image, 15)
        if H == 0:
            image_ = GetRed(image, 14)
            Pn, Pa = compute_Pn_Pa(image_)

    kernel_size = 1  # 可以根据需要调整
    image_ = skimage.morphology.closing(image_)
    image_ = skimage.morphology.opening(image_, skimage.morphology.disk(kernel_size))
    image_ = skimage.filters.gaussian(image_, sigma=0.1)

    # 提取边缘特征
    edges = skimage.feature.canny(image_)

    return image_, H, Pn, Pa, edges, gray_features



def check_initial_file_order(self):
    print("Initial file list order:")
    for i, filename in enumerate(os.listdir(self.root_dir)):
        if i < 10:  # 只打印前10个文件
            print(f"{i}: {filename}")


import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # 如果没有提供alpha，根据类别数量自动计算
        if alpha is None:
            self.alpha = torch.ones(num_classes) / num_classes
        else:
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        # inputs: [N, C], targets: [N]
        N, C = inputs.shape

        # 计算log_softmax
        log_softmax = F.log_softmax(inputs, dim=1)

        # 获取目标类别的log概率
        targets_one_hot = F.one_hot(targets, C)
        log_pt = torch.sum(log_softmax * targets_one_hot, dim=1)

        # 计算pt
        pt = torch.exp(log_pt)

        # 获取对应类别的alpha值
        batch_alpha = self.alpha[targets].to(inputs.device)

        # 计算focal loss
        focal_loss = -batch_alpha * (1 - pt) ** self.gamma * log_pt

        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []
    num_classes = len(data_loader.dataset.label_to_idx)
    class_correct = torch.zeros(num_classes).to(device)
    class_total = torch.zeros(num_classes).to(device)

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

            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=1
    )
    cm = confusion_matrix(all_labels, all_predictions)

    # Calculate per-class accuracy
    class_acc = class_correct / class_total

    return total_loss / len(
        data_loader), precision, recall, f1, cm, all_labels, all_probs, accuracy, class_acc.cpu().numpy()
