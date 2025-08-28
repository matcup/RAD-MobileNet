import os
import re
import logging
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from src.utilities import adopt_HSV_H_
import configparser
import torch.nn as nn
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
config = configparser.ConfigParser()
config.read('C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\config\config.ini')
feature_ = int(config['train']['feature'])

class DualAttentionReduction(nn.Module):
    def __init__(self):
        super().__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 最终降维
        self.final_conv = nn.Conv2d(4, 3, 1)

    def forward(self, x):
        # 通道注意力
        chan_att = self.channel_attention(x)
        x = x * chan_att

        # 空间注意力
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att

        # 通道降维
        return self.final_conv(x)


class SealDataset(Dataset):
    def __init__(self, root_dir, transform=None, all_labels=None, augmentation=None, feature_=0):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_to_idx = {}
        self.augmentation = augmentation
        self.feature_ = feature_

        # 初始化卷积降维层（feature_=3）
        self.conv_reduction = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        with torch.no_grad():
            self.conv_reduction.weight.data[:3, :3, :, :] = torch.eye(3).view(3, 3, 1, 1)
            self.conv_reduction.weight.data[:, 3:, :, :] = 0.1
            self.conv_reduction.bias.data.zero_()

        # 初始化注意力降维层（feature_=4）
        self.attention_reduction = DualAttentionReduction()

        if all_labels:
            self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                label = self.extract_label(filename)

                if label not in self.label_to_idx:
                    if all_labels:
                        continue
                    else:
                        self.label_to_idx[label] = len(self.label_to_idx)

                self.images.append(os.path.join(root_dir, filename))
                self.labels.append(self.label_to_idx[label])

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # 打印标签映射信息
        logging.info("Label to index mapping:")
        for label, idx in self.label_to_idx.items():
            logging.info(f"{label}: {idx}")

        # 验证所有标签
        invalid_labels = [label for label in self.labels if not (0 <= label < len(self.label_to_idx))]
        if invalid_labels:
            raise ValueError(f"Found invalid labels: {invalid_labels}")

        logging.info(f"Loaded {len(self.images)} images with {len(self.label_to_idx)} unique labels.")

    def extract_label(self, filename):
        name_without_ext = os.path.splitext(filename)[0]
        if '@' in name_without_ext:
            label = name_without_ext.split('@')[0].strip()
        else:
            pattern = r'^(.+?)\d*$'
            match = re.match(pattern, name_without_ext)
            label = match.group(1).rstrip() if match else name_without_ext
        return label

    def process_image_with_feature(self, image_np, feature_):
        # 如果是原始图像模式，直接返回
        if feature_ == 0:
            return image_np / 255.0 if image_np.max() > 1 else image_np

        # 只有在需要特征时才调用adopt_HSV_H_
        image_, _, _, _, edges, gray_features = adopt_HSV_H_(image_np)

        # 归一化
        image_np = image_np / 255.0 if image_np.max() > 1 else image_np
        image_ = image_ / 255.0 if image_.max() > 1 else image_

        # 将特征图扩展为3D
        image_ = np.expand_dims(image_, axis=-1)

        # 四通道合并
        combined_image = np.concatenate([image_np, image_], axis=-1)  # (H, W, 4)

        if feature_ == 1:  # 第四通道作为权重相乘
            rgb = combined_image[:, :, :3]
            mask = combined_image[:, :, 3:]
            return rgb * mask

        elif feature_ == 2:  # 第四通道作为权重相加
            rgb = combined_image[:, :, :3]
            mask = combined_image[:, :, 3:]
            return rgb + mask

        elif feature_ == 3:  # 使用卷积降维
            x = torch.from_numpy(combined_image.transpose(2, 0, 1)).float()
            x = x.unsqueeze(0)
            with torch.no_grad():
                output = self.conv_reduction(x)
            return output.squeeze(0).numpy().transpose(1, 2, 0)

        elif feature_ == 4:  # 使用注意力机制降维
            x = torch.from_numpy(combined_image.transpose(2, 0, 1)).float()
            x = x.unsqueeze(0)
            with torch.no_grad():
                output = self.attention_reduction(x)
            return output.squeeze(0).numpy().transpose(1, 2, 0)

        else:
            raise ValueError(f"Unsupported feature_ value: {feature_}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # 添加标签验证
        num_classes = len(self.label_to_idx)
        if not (0 <= label < num_classes):
            logging.error(
                f"Invalid label {label} for image {img_path}. Label should be in range [0, {num_classes - 1}]")
            label = 0

        try:
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)

            processed_image = self.process_image_with_feature(image_np, self.feature_)
            processed_image = np.clip(processed_image * 255, 0, 255).astype(np.uint8)
            processed_image = Image.fromarray(processed_image)

            if self.augmentation:
                processed_image = np.array(processed_image)
                processed_image = self.augmentation(processed_image)
                processed_image = Image.fromarray(processed_image.astype('uint8'))

            if self.transform:
                processed_image = self.transform(processed_image)

            return processed_image, label

        except Exception as e:
            logging.error(f"Error processing image {img_path}: {str(e)}")
            placeholder_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                placeholder_image = self.transform(placeholder_image)
            return placeholder_image, label