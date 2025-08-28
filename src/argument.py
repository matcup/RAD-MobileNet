"""
1    2   3  4       5       6        7   8   9        10      11      12
原图	缩小	剪裁	亮度降低 对比度增强	图像掩盖	旋转	翻转 字迹覆盖	纹理覆盖	椒盐噪声	边缘膨胀
"""
import os
import cv2
import numpy as np
import random
import re
from collections import defaultdict
from skimage import morphology, exposure
from PIL import Image, ImageDraw, ImageFont
from skimage.util import random_noise


def extract_label_a(filename):
    label = os.path.splitext(filename)[0]
    label = re.sub(r'\d+$', '', label)
    return label


def image_downscale(image):
    r = random.uniform(3, 5)
    h, w = image.shape[:2]
    new_h, new_w = int(h / r), int(w / r)
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def image_crop(image):
    r1 = random.random()
    r2 = random.uniform(0.1, 0.4)  # 裁剪 10% 到 60%
    h, w = image.shape[:2]

    if r1 < 0.5:  # horizontal crop
        crop_w = int(w * (1 - r2))  # 保留的宽度
        if random.random() < 0.5:
            # 从左边裁剪
            return image[:, w - crop_w:]
        else:
            # 从右边裁剪
            return image[:, :crop_w]
    else:  # vertical crop
        crop_h = int(h * (1 - r2))  # 保留的高度
        if random.random() < 0.5:
            # 从上边裁剪
            return image[h - crop_h:, :]
        else:
            # 从下边裁剪
            return image[:crop_h, :]


def adjust_brightness(image):
    r = random.uniform(0.7, 4)
    return exposure.adjust_gamma(image, r)


def adjust_contrast(image):
    r = random.uniform(0.4, 1.4)
    return exposure.adjust_gamma(image, r)


def image_mask(image):
    num_masks = random.randint(1, 10)
    h, w = image.shape[:2]

    # 创建一个与原图像大小相同的掩码数组
    mask = np.ones_like(image, dtype=bool)

    for _ in range(num_masks):
        # 计算掩码尺寸为图像尺寸的 10%-20%
        mask_h = int(h * random.uniform(0.05, 0.1))
        mask_w = int(w * random.uniform(0.05, 0.1))

        # 确保掩码位置不会超出图像边界
        x = random.randint(0, w - mask_w)
        y = random.randint(0, h - mask_h)

        # 在掩码数组中标记要遮盖的区域
        mask[y:y + mask_h, x:x + mask_w] = False

    # 一次性应用掩码到原图像
    masked_image = image.copy()
    masked_image[~mask] = 0

    return masked_image


def image_rotate(image):
    angle = random.randint(-30, 30)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))


def image_flip(image):
    flipped = cv2.flip(image, 1)
    return (flipped)


def text_overlay(image):
    # This is a simplified version. You'll need to implement the full algorithm
    font = ImageFont.truetype("C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\data\cocacola.ttf", random.randint(90, 230))
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    text = random.choice(
        "永和九年岁在癸丑暮春之初会于稽山阴兰亭修禊事也群贤毕至少长咸集此地崇峻岭茂林竹又清\流激湍映带左右引以为觞曲水列坐次虽无丝管弦盛一咏亦足畅叙\
        幽情是日天朗气惠风和仰观宇宙大俯察品类游目骋怀极视听娱信可乐夫人相与俯仰世或取诸怀抱悟言室内因寄所托放浪形骸之外舍万殊静躁不同当欣遇暂得快然\
        自知老将及倦随迁感慨系向已陈迹犹能兴况短化终期尽古云死生亦痛哉每览昔兴若合契未尝临文嗟悼喻固知虚诞齐彭殇妄作后视今犹悲故列叙时录述殊异致览\
        者亦将有")
    position = (random.randint(0, image.shape[1]), random.randint(0, image.shape[0]))
    draw.text(position, text, font=font, fill=(0, 0, 0))
    return np.array(img_pil)


def texture_overlay(image):
    # 指定纹理图像所在的目录
    texture_dir = r"C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\data\wenli1"

    # 获取目录中所有的 PNG 文件
    texture_files = [f for f in os.listdir(texture_dir) if f.lower().endswith('.png')]

    if not texture_files:
        print("没有找到 PNG 纹理图像")
        return image

    # 随机选择一个纹理图像
    texture_file = random.choice(texture_files)
    texture_path = os.path.join(texture_dir, texture_file)

    # 打开纹理图像
    texture = Image.open(texture_path).convert("RGBA")

    # 将输入图像转换为 PIL Image 对象
    img_pil = Image.fromarray(image).convert("RGBA")

    # 随机调整纹理大小（在原始尺寸的 50% 到 150% 之间）
    scale = random.uniform(1, 3)
    new_size = (int(texture.width * scale), int(texture.height * scale))
    texture = texture.resize(new_size, Image.LANCZOS)

    # 随机选择纹理的位置
    x = random.randint(0, max(0, img_pil.width - texture.width))
    y = random.randint(0, max(0, img_pil.height - texture.height))

    # 创建一个与原图像大小相同的透明图层
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))

    # 将纹理图像粘贴到这个图层上的随机位置
    overlay.paste(texture, (x, y), texture)

    # 随机设置纹理的不透明度
    opacity = random.uniform(0.1, 0.4)
    overlay = Image.blend(Image.new('RGBA', img_pil.size, (0, 0, 0, 0)), overlay, opacity)

    # 将纹理图层与原图像合并
    result = Image.alpha_composite(img_pil, overlay)

    # 转换回 numpy 数组并返回
    return np.array(result.convert('RGB'))


def add_noise(image):
    noise_type = random.choice(['gaussian', 'salt', 'pepper', 's&p'])

    # 将图像转换为浮点型，范围在 0-1 之间
    image_float = image.astype('float32') / 255.0

    # 添加噪声
    noisy_image = random_noise(image_float, mode=noise_type)

    # 将图像转回 uint8 类型，范围在 0-255 之间
    noisy_image = (noisy_image * 255).astype(np.uint8)

    return noisy_image


def edge_dilation(image):
    # Convert to HSV and extract red regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Dilate the mask
    r = random.randint(1, 7)
    kernel = np.ones((r, r), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)

    # Apply the dilated mask to the original image
    result = image.copy()
    result[dilated == 255] = image[mask == 255].mean(axis=0)

    return result


def no_change(image):

    return image

ARGUMENT = 'image_mask'


# 处理图像并记录成功处理的文件
processed_files = set()

# 定义图像处理函数字典
image_processing_functions = {
    # 'image_downscale': image_downscale,
    # 'image_crop': image_crop,
    # 'adjust_brightness': adjust_brightness,
    # 'adjust_contrast': adjust_contrast,
    # 'image_mask': image_mask,
    # 'image_rotate': image_rotate,
    # 'image_flip': image_flip,
    # 'text_overlay': text_overlay,
    # 'texture_overlay': texture_overlay,
    # 'add_noise': add_noise,
    # 'edge_dilation': edge_dilation,
    'no_change' : no_change
}


def apply_multiple_augmentations(image, num_augmentations):
    """
    随机选择并应用多个数据增强模块
    """
    augmentation_sequence = random.sample(list(image_processing_functions.keys()), num_augmentations)

    for augmentation in augmentation_sequence:
        image = image_processing_functions[augmentation](image)

    return image


folder_a = r'C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\data\train'
folder_b = r'C:\Work\PycharmProjects\Seal_edge_recognize\Chinese-Seal-Recognize\data\argument_finalc'

os.makedirs(folder_b, exist_ok=True)

# 用于跟踪每个标签的计数
label_counter = defaultdict(int)

# 获取原始文件夹中的所有文件名
original_files = [f for f in os.listdir(folder_a) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 处理图像并记录成功处理的文件
processed_files = set()

for filename in original_files:
    label = extract_label_a(filename)
    img_path = os.path.join(folder_a, filename)

    try:
        original_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if original_img is None:
            raise Exception("Image is None")
    except Exception as e:
        print(f"Error reading image {filename}: {str(e)}")
        continue

    success_count = 0
    for i in range(1):  # 为每个原始图像生成3张增强图像
        # 使用多个随机选择的图像处理函数
        processed_img = apply_multiple_augmentations(original_img, num_augmentations=1)

        # 生成唯一的文件名
        while True:
            label_counter[label] += 1
            new_filename = f"{label}@{label_counter[label]}.png"
            new_path = os.path.join(folder_b, new_filename)
            if not os.path.exists(new_path):
                break

        is_success, im_buf_arr = cv2.imencode(".png", processed_img)
        if is_success:
            try:
                im_buf_arr.tofile(new_path)
                success_count += 1
            except Exception as e:
                print(f"Error saving image {new_filename}: {str(e)}")
        else:
            print(f"Failed to encode image: {new_filename}")

    if success_count == 3:
        processed_files.add(filename)

# 找出未被处理的文件
unprocessed_files = set(original_files) - processed_files

print("\nProcessing Summary:")
print(f"Total original images: {len(original_files)}")
print(f"Successfully processed images: {len(processed_files)}")
print(f"Unprocessed images: {len(unprocessed_files)}")

if unprocessed_files:
    print("\nUnprocessed Images:")
    for img in sorted(unprocessed_files):
        print(img)
else:
    print("\nAll images were processed successfully.")

print("\nProcessing complete.")