import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
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

def process_image_with_mask(img_np, mask, model):
    # 确保图像为 uint8 类型
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)

    # 确保图像为 3 通道
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # 使用 np.nonzero 获取掩码中非零元素的坐标
    y, x = np.nonzero(mask)

    if len(y) == 0 or len(x) == 0:
        print("警告: 掩码为空。没有区域需要增强。")
        return img_np

    # 获取掩码边界
    top, bottom, left, right = y.min(), y.max(), x.min(), x.max()

    # 裁剪感兴趣区域
    roi = img_np[top:bottom + 1, left:right + 1]

    # 使用 Real-ESRGAN 处理 ROI
    enhanced_roi, _ = model.enhance(roi, outscale=1)  # 将 outscale 设置为 1，保持原始大小

    # 创建结果图像
    result = img_np.copy()

    # 将增强的 ROI 应用到结果图像上，使用掩码
    mask_roi = mask[top:bottom + 1, left:right + 1]
    mask_roi = np.repeat(mask_roi[:, :, np.newaxis], 3, axis=2)
    result[top:bottom + 1, left:right + 1] = np.where(mask_roi, enhanced_roi, result[top:bottom + 1, left:right + 1])

    return result

# 设置输入图像路径
input_path = 'C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\data\\1.png'  # 替换为您的图像路径

# 读取原始图像
original_img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), 1)

# 预处理图像
x_, H, Pn, Pa = adopt_HSV_H(original_img)

print("原始图像形状:", original_img.shape)
print("预处理后图像形状:", x_.shape)
print("预处理后图像数据类型:", x_.dtype)
print("预处理后图像值范围:", x_.min(), "到", x_.max())

# 创建掩码
x_uint8 = (x_ * 255).astype(np.uint8)
_, mask = cv2.threshold(x_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
mask = mask > 0

print("掩码形状:", mask.shape)
print("掩码数据类型:", mask.dtype)
print("掩码唯一值:", np.unique(mask))

# 如果掩码为空，尝试反转
if not np.any(mask):
    mask = ~mask
    print("掩码已反转")
    print("反转后掩码唯一值:", np.unique(mask))

# 初始化模型
model = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True,
)

# 处理图像
print('正在处理图像...')
enhanced_img = process_image_with_mask(original_img, mask, model)

print("增强后图像形状:", enhanced_img.shape)

# 显示结果
plt.figure(figsize=(20, 15))

plt.subplot(2, 2, 1)
plt.title('原始图像')
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('掩码')
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('增强图像')
plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('差异')
diff = cv2.subtract(enhanced_img, original_img)
plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存增强后的图像
output_path = os.path.join(os.path.dirname(input_path), 'enhanced_' + os.path.basename(input_path))
cv2.imwrite(output_path, enhanced_img)

print('图像处理完成。增强后的图像已保存至:', output_path)