from torchvision import models, transforms

# def get_transform(image_size=224):
#     return transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])



def get_transform(model_name):
    if model_name in ['efficientnet_b0', 'vit_b_16', 'deit_small_patch16_224']:
        image_size = 224
    elif model_name == 'inception_v3':
        image_size = 299
    else:
        image_size = 224  # 默认大小，可以根据需要调整

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

