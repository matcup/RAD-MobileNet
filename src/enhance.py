import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import morphology, exposure
import random
from skimage.util import random_noise


class SealDataAugmentation:
    def __init__(self, method_indices=None):
        self.all_augmentations = [
            self.image_downscale,
            self.image_crop,
            self.adjust_brightness,
            self.adjust_contrast,
            self.image_mask,
            self.image_rotate,
            self.image_flip,
            self.text_overlay,
            self.texture_overlay,
            self.add_noise,
            self.edge_dilation
        ]

        if method_indices is None or -1 in method_indices:
            self.augmentations = self.all_augmentations
        elif 0 in method_indices:
            self.augmentations = []
        else:
            self.augmentations = [self.all_augmentations[i - 1] for i in method_indices if
                                  1 <= i <= len(self.all_augmentations)]

    def __call__(self, image):
        if not self.augmentations:
            return image

        # 随机选择1到3个增强方法应用（如果可用的方法少于3个，则选择所有可用方法）
        num_augmentations = random.randint(1, min(3, len(self.augmentations)))
        selected_methods = random.sample(self.augmentations, num_augmentations)

        for method in selected_methods:
            image = method(image)

        return image

    # 其余方法保持不变
    def image_downscale(self, image):
        r = random.uniform(1, 3)
        h, w = image.shape[:2]
        new_h, new_w = int(h / r), int(w / r)
        small = cv2.resize(image, (new_w, new_h))
        return cv2.resize(small, (w, h))

    def image_crop(self, image):
        r1, r2 = random.random(), random.uniform(0.1, 0.6)
        h, w = image.shape[:2]
        if r1 < 0.5:  # horizontal crop
            crop_w = int(w * r2)
            start = 0 if int(r1 * 10) % 2 == 0 else w - crop_w
            return image[:, start:start + crop_w]
        else:  # vertical crop
            crop_h = int(h * r2)
            start = 0 if int(r1 * 10) % 2 == 0 else h - crop_h
            return image[start:start + crop_h, :]

    def adjust_brightness(self, image):
        r = random.uniform(0.2, 4)
        return exposure.adjust_gamma(image, r)

    def adjust_contrast(self, image):
        r = random.uniform(0.2, 1.2)
        return exposure.adjust_contrast(image, r)

    def image_mask(self, image):
        num_masks = random.randint(1, 10)
        h, w = image.shape[:2]
        for _ in range(num_masks):
            mask_size = random.randint(5, 50)
            x = random.randint(0, w - mask_size)
            y = random.randint(0, h - mask_size)
            image[y:y + mask_size, x:x + mask_size] = 0
        return image

    def image_rotate(self, image):
        angle = random.randint(-30, 30)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))

    def image_flip(self, image):
        flipped = cv2.flip(image, 1)
        return self.image_rotate(flipped)

    def text_overlay(self, image):
        # This is a simplified version. You'll need to implement the full algorithm
        font = ImageFont.truetype("path/to/handwritten/font.ttf", random.randint(5, 30))
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        text = random.choice("兰亭序")
        position = (random.randint(0, image.shape[1]), random.randint(0, image.shape[0]))
        draw.text(position, text, font=font, fill=(0, 0, 0))
        return np.array(img_pil)

    def texture_overlay(self, image):
        # You'll need to implement this based on your texture images
        pass

    def add_noise(self, image):
        noise_type = random.choice(['gaussian', 'salt', 'pepper', 's&p'])

        # 将图像转换为浮点型，范围在 0-1 之间
        image_float = image.astype('float32') / 255.0

        # 添加噪声
        noisy_image = random_noise(image_float, mode=noise_type)

        # 将图像转回 uint8 类型，范围在 0-255 之间
        noisy_image = (noisy_image * 255).astype(np.uint8)

        return noisy_image

    def edge_dilation(self, image):
        # Convert to HSV and extract red regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Dilate the mask
        r = random.randint(5, 15)
        kernel = np.ones((r, r), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # Apply the dilated mask to the original image
        result = image.copy()
        result[dilated == 255] = image[mask == 255].mean(axis=0)

        return result