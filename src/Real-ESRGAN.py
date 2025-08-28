import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os
import sys
import cv2

from utilities import *


def process_image(img_np, model):
    # 确保图像是uint8类型
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)

    # 确保图像是3通道的
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # 使用Real-ESRGAN处理图像
    output, _ = model.enhance(img_np, outscale=4)

    return output


# 设置输入图像路径
input_path = 'C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\data/3.png'  # 请替换为您的图像路径
input_path = 'C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\data\\test\乾隆-三希堂A1.png'
# 读取并预处理图像
x_ = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), 1)
# x_, H, Pn, Pa = adopt_HSV_H(x_)

# 初始化模型
model = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
)

# 处理图像
print('Processing image...')
enhanced_img = process_image(x_, model)

# 显示结果
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.title('Preprocessed Image')
plt.imshow(x_, cmap='gray')  # 使用灰度模式显示
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Enhanced Image')
plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))  # 转换颜色空间以正确显示
plt.axis('off')

plt.tight_layout()
plt.show()

print('Image processing completed.')