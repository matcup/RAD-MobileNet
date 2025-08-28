# RAD-MobileNet
RAD-MobileNet: A lightweight framework for Chinese seal recognition that integrates feature fusion for background noise reduction, dilated convolutions for expanded receptive fields, and targeted data augmentation strategies to address data scarcity. 

This document provides comprehensive instructions for using the RAD-MobileNet framework for Chinese seal recognition.

## Table of Contents

1. Configuration
2. Training
3. Evaluation
4. Model Architecture
5. Troubleshooting

## Configuration

The system is configured through the `config/config.ini` file, which contains the following key sections:

### Data Paths

```ini
[data_path]
train_path = path/to/train/data
val_path = path/to/validation/data
test_path = path/to/test/data
```

### Model Selection

```ini
[models]
# Available models: CNN, vgg16, vgg19, resnet50, resnet101, resnet152, 
# inception_v3, efficientnet, vit, MobileNetV2, MobileNetV3, 
# ShuffleNet, SqueezeNet, DilatedMobileNetV2
name = DilatedMobileNetV2
```

### Training Parameters

```ini
[train]
batch_size = 16
num_epochs = 100
lr = 0.001
weight_decay = 1e-4
feature = 4         # Feature mode: 3 for RGB only, 4 for RGB+Seal Channel
model_path = path/to/save/models
```

### Early Stopping

```ini
[early_stop]
patience = 5        # Number of epochs with no improvement after which training stops
```

### Results Visualization

```ini
[results]
cm_show = 1         # 0=show confusion matrix, 1=hide
```

### Data Augmentation

```ini
[data_augmentation]
method_indices = [1, 2, 3]  # Indices of augmentation methods to use
```

## Training

To train the RAD-MobileNet model:

1. Configure your settings in `config/config.ini`

2. Run the training script:

   ```vim
   python scripts/train.py
   ```

The training process will:

- Load and preprocess the dataset
- Initialize the selected model architecture
- Train using the specified parameters
- Apply early stopping based on validation F1 score
- Save the best model to the specified path

### Feature Modes

The framework supports two primary feature extraction modes:

- `feature = 3`: Standard RGB channel processing
- `feature = 4`: Enhanced RGB+Seal Channel processing (recommended for improved performance)

## Evaluation

The model is automatically evaluated on the test set after training. Performance metrics include:

- Loss
- Precision
- Recall
- F1 Score
- Confusion Matrix (optional display)

## Model Architecture

RAD-MobileNet consists of three key components:

1. **Seal Channel Extraction**: Extracts the red seal region using HSV color space optimization.
2. **Feature Fusion Module**: Combines RGB channels with seal-specific information through a dual attention mechanism.
3. **Dilated MobileNetV2 Backbone**: Incorporates dilated convolutions to expand the receptive field while maintaining computational efficiency.

To use different components of the architecture:

```python
# For seal channel extraction only
from src.utilities import extract_seal_channel
seal_channel = extract_seal_channel(image)

# For complete RAD-MobileNet with feature fusion
from src.model import DilatedMobileNetV2
model = DilatedMobileNetV2(num_classes=YOUR_NUM_CLASSES, feature_mode=4)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config.ini
   - Try a more memory-efficient model variant
2. **Poor Performance on Test Set**
   - Increase data augmentation complexity
   - Adjust learning rate or weight decay
   - Try different feature modes (3 vs 4)
3. **Path Errors**
   - Ensure all paths in config.ini are absolute paths
   - Check for correct directory structure
