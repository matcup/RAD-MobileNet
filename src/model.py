import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import VGG16_Weights, VGG19_Weights

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def vgg16(num_classes):
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    if num_classes != 1000:
        model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def vgg19(num_classes):
    model = models.vgg19(weights=VGG19_Weights.DEFAULT)
    if num_classes != 1000:
        model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def resnet50(num_classes):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def resnet101(num_classes):
    model = models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def resnet152(num_classes):
    model = models.resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def inception_v3(num_classes):
    model = models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    model.aux_logits = False  # 禁用辅助输出
    return model


def efficientnet(num_classes):
    model = models.efficientnet_b0(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, num_classes)
    )
    return model


def vit(num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)
    return model


def vit_tiny(num_classes, dropout_rate=0.3, pretrained=True):
    model_path = "/data01/zhangzj/project/Seal_edge_recognize/model/vit_tiny_patch16_224"

    # 创建模型
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)

    if pretrained:
        # 检查是否存在 safetensors 文件
        safetensors_path = os.path.join(model_path, "model.safetensors")
        bin_path = os.path.join(model_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path)
        else:
            raise FileNotFoundError(f"No model weights found in {model_path}")

        model.load_state_dict(state_dict)

    # 冻结预训练参数
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # 获取输入特征数
    num_features = model.head.in_features

    # 替换分类头部，添加dropout
    model.head = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes)
    )

    return model


class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeNet, self).__init__()
        self.model = models.squeezenet1_1(pretrained=True)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


class ShuffleNet(nn.Module):
    def __init__(self, num_classes):
        super(ShuffleNet, self).__init__()
        self.model = models.shufflenet_v2_x1_0(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class DilatedInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(DilatedInvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ))

        layers.extend([
            # dw with dilation
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=dilation,
                      dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class DilatedMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(DilatedMobileNetV2, self).__init__()
        # 加载预训练模型
        base_model = models.mobilenet_v2(pretrained=True)

        # 提取特征提取器部分
        features = list(base_model.features.children())

        # 修改后半部分的模块，添加空洞卷积
        # 这里我们修改最后几个块的空洞率
        dilations = [2, 4, 8]  # 不同层使用不同的空洞率
        dilation_idx = 0

        modified_features = []
        for i, feature in enumerate(features):
            if i < len(features) - 3:  # 保持前面层不变
                modified_features.append(feature)
            else:  # 修改最后几层
                if isinstance(feature, nn.Sequential):
                    inv_res = feature[0]  # 获取InvertedResidual模块
                    # 获取输入和输出通道数
                    if hasattr(inv_res, 'conv'):
                        in_channels = inv_res.conv[0][0].in_channels
                        out_channels = inv_res.conv[-1][0].out_channels
                    else:
                        # 使用alternative方式获取通道数
                        in_channels = inv_res.in_channels if hasattr(inv_res, 'in_channels') else inv_res._in_channels
                        out_channels = inv_res.out_channels if hasattr(inv_res,
                                                                       'out_channels') else inv_res._out_channels

                    # 创建新的带空洞的模块
                    new_inv_res = DilatedInvertedResidual(
                        in_channels,
                        out_channels,
                        inv_res.stride,
                        expand_ratio=6,  # MobileNetV2默认的扩展率
                        dilation=dilations[dilation_idx]
                    )
                    modified_features.append(new_inv_res)
                    dilation_idx = (dilation_idx + 1) % len(dilations)
                else:
                    modified_features.append(feature)

        # 重建模型
        self.features = nn.Sequential(*modified_features)

        # 分类器部分
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

        # 初始化新添加的层
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



# model = DilatedMobileNetV2(num_classes)