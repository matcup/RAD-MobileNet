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
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.morphology import skeletonize
from skimage.filters import gabor
from scipy.fftpack import fft2
from scipy.stats import kurtosis, skew
# 设置输入图像路径
input_path = r''  # 替换为您的输入图像路径


def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. 边缘提取
    canny = cv2.Canny(gray, 100, 200)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 2. 形状描述符
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        extent = float(area) / (w * h) if w * h > 0 else 0
    else:
        circularity, aspect_ratio, extent = 0, 0, 0

    # 3. 纹理分析
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    gabor_real, gabor_imag = gabor(gray, frequency=0.6)

    # 4. 颜色特征
    hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    # 5. 局部特征 (SIFT)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 6. 笔画分析 (简化版)
    horizontal_lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    vertical_lines = cv2.HoughLinesP(binary, 1, np.pi / 2, 100, minLineLength=100, maxLineGap=10)
    diagonal_lines = cv2.HoughLinesP(binary, 1, np.pi / 4, 100, minLineLength=100, maxLineGap=10)
    stroke_features = np.array(
        [len(l) if l is not None else 0 for l in [horizontal_lines, vertical_lines, diagonal_lines]])

    # 7. 密度分布
    density_map = cv2.resize(gray, (8, 8))
    density_features = density_map.flatten() / 255.0

    # 8. 傅里叶描述子
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # 添加1以避免log(0)
    fourier_features = magnitude_spectrum.flatten()[:100]  # 取前100个特征

    # 9. 骨架提取
    skeleton = skeletonize(binary)

    # 10. 深度特征 (这里需要一个预训练的模型，这里省略)

    # 确保所有特征都是一维数组
    shape_features = np.array([circularity, aspect_ratio, extent])
    lbp_features = np.array([np.mean(lbp), np.std(lbp)])
    gabor_features = np.array([np.mean(gabor_real), np.std(gabor_real),
                               np.mean(gabor_imag), np.std(gabor_imag)])
    fourier_features = fourier_features[:10]  # 取前10个傅里叶特征
    skeleton_feature = np.array([np.sum(skeleton) / skeleton.size])  # 骨架密度

    # 打印每个特征的形状，以便调试
    print("Shape features shape:", shape_features.shape)
    print("HSV histogram shape:", hsv_hist.shape)
    print("LBP features shape:", lbp_features.shape)
    print("Gabor features shape:", gabor_features.shape)
    print("Stroke features shape:", stroke_features.shape)
    print("Density features shape:", density_features.shape)
    print("Fourier features shape:", fourier_features.shape)
    print("Skeleton feature shape:", skeleton_feature.shape)

    # 组合所有特征
    features = np.concatenate([
        shape_features,
        hsv_hist,
        lbp_features,
        gabor_features,
        stroke_features,
        density_features,
        fourier_features,
        skeleton_feature
    ])

    return features, canny, sobel, lbp, gabor_real, skeleton


# ... [其余代码保持不变] ...


# 读取并预处理图像
x = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)

x_, H, Pn, Pa, edges, gray_features = adopt_HSV_H_(x, 350)

# 提取新的特征
features, canny, sobel, lbp, gabor, skeleton = extract_features(x)

# 创建一个更大的图形，包含更多子图
plt.figure(figsize=(20, 20))

# 显示原始图像和处理后的图像
plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(x_, cmap='gray')
plt.title('Processed Image')
plt.axis('off')

# 显示Canny边缘
plt.subplot(3, 3, 3)
plt.imshow(canny, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

# 显示Sobel边缘
plt.subplot(3, 3, 4)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel Edges')
plt.axis('off')

# 显示LBP纹理
plt.subplot(3, 3, 5)
plt.imshow(lbp, cmap='gray')
plt.title('LBP Texture')
plt.axis('off')

# 显示Gabor滤波结果
plt.subplot(3, 3, 6)
plt.imshow(gabor, cmap='gray')
plt.title('Gabor Filter')
plt.axis('off')

# 显示骨架
plt.subplot(3, 3, 7)
plt.imshow(skeleton, cmap='gray')
plt.title('Skeleton')
plt.axis('off')

# 调整子图之间的间距
plt.tight_layout()

# 显示图像
plt.show()

# 打印特征向量
print("\nFeature Vector:")
print(features)

print('Image has been processed and features have been extracted.')