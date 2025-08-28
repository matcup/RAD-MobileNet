import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到Python路径
sys.path.append(project_root)
from src.utilities import *

# 设置输入和输出目录
input_dir = r'C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\data\data_n\test'  # 替换为您的输入目录路径
output_dir = r'C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\data\data_n\test\output'  # 替换为您的输出目录路径

# 如果输出目录不存在，创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 处理目录中的所有图像
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 可以根据需要添加其他图像格式
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取并预处理图像
        x_ = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        x_, H, Pn, Pa = adopt_HSV_H(x_)

        print(f"Processing {filename}: H={H}, Pn={Pn}, Pa={Pa}")

        # 将预处理后的图像保存到输出目录
        plt.imsave(output_path, x_, cmap='gray')

        # 使用 cv2 保存图像（如果 plt.imsave 仍然出现问题）
        # cv2.imwrite(output_path, (x_ * 255).astype(np.uint8))

print('All images have been processed and saved.')