"""
Neural Network Models for FL-NILM
Supports multiple architectures: CNN, ResNet, MobileNet, VGG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleCNN(nn.Module):
    """Simple CNN architecture"""
    
    def __init__(self, num_classes, img_size=192, conv_filters=[32, 32, 32, 32], 
                 fc_units=128, dropout=0.5):
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, conv_filters[0], kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(4, 4)  # 192 -> 48
        
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 48 -> 24
        
        self.conv3 = nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 24 -> 12
        
        self.conv4 = nn.Conv2d(conv_filters[2], conv_filters[3], kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # 12 -> 6
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_filters[3] * 6 * 6, fc_units)
        self.fc2 = nn.Linear(fc_units, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class SimpleCNN_Small(nn.Module):
    def __init__(self, num_classes, input_size=192):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(4, 4),
            nn.Conv2d(4, 4, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(4, 4),
            nn.Conv2d(4, 16, 3, 1, 1), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, 1, 1), nn.ReLU(),
        )
        
        # Adaptive pooling collapses any spatial size to 1x1.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # After adaptive pooling the flattened feature size is always 4.
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)  # Collapse to 1x1.
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleCNN_Medium(nn.Module):
    """
    Medium-sized CNN - Between full CNN (Teacher) and CNN_Small (Student)
    
    Architecture:
    - 4 conv layers with [16, 16, 24, 16] filters
    - FC layer with 64 units
    - Moderate dropout (0.3)
    
    Parameter comparison:
    - SimpleCNN (Teacher): ~176K params
    - SimpleCNN_Medium: ~20K params (11% of Teacher)
    - SimpleCNN_Small (Student): ~1.5K params (0.8% of Teacher)
    
    Use cases:
    - Alternative Student model for OLFED (more capacity than cnn_small)
    - Standalone model for resource-constrained scenarios
    - Intermediate model for progressive training
    """
    
    def __init__(self, num_classes, input_size=192, dropout=0.3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers with moderate filter sizes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(4, 4)  # 192 -> 48
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 48 -> 24
        
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 24 -> 12
        
        self.conv4 = nn.Conv2d(24, 16, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # 12 -> 6
        
        # Fully connected layers with moderate capacity
        self.fc1 = nn.Linear(16 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ResNet18NILM(nn.Module):
    """ResNet18 adapted for single-channel NILM images"""
    
    def __init__(self, num_classes, pretrained=False):
        super(ResNet18NILM, self).__init__()
        
        self.num_classes = num_classes
        
        # Load ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for single channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, 
                                      padding=3, bias=False)
        
        # Modify final FC layer for our number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)


class MobileNetV2NILM(nn.Module):
    """MobileNetV2 adapted for NILM"""
    
    def __init__(self, num_classes, width_mult=1.0):
        super(MobileNetV2NILM, self).__init__()
        
        self.num_classes = num_classes
        
        # Load MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=False, width_mult=width_mult)
        
        # Modify first conv for single channel
        self.mobilenet.features[0][0] = nn.Conv2d(1, int(32 * width_mult), 
                                                   kernel_size=3, stride=2, 
                                                   padding=1, bias=False)
        
        # Modify classifier
        self.mobilenet.classifier[1] = nn.Linear(
            self.mobilenet.last_channel, num_classes
        )
        
    def forward(self, x):
        return self.mobilenet(x)


class VGG16NILM(nn.Module):
    """VGG16 adapted for NILM"""
    
    def __init__(self, num_classes, batch_norm=True):
        super(VGG16NILM, self).__init__()
        
        self.num_classes = num_classes
        
        # Load VGG16
        if batch_norm:
            self.vgg = models.vgg16_bn(pretrained=False)
        else:
            self.vgg = models.vgg16(pretrained=False)
        
        # Modify first conv for single channel
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Modify classifier
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        return self.vgg(x)


def create_model(model_name, num_classes, config_dict=None):
    """
    Factory function to create models
    
    Args:
        model_name: Name of the model ('cnn', 'resnet18', 'mobilenet', 'vgg16')
        num_classes: Number of output classes
        config_dict: Optional configuration dictionary
    
    Returns:
        Model instance
    """
    config_dict = config_dict or {}
    
    if model_name == 'cnn':
        return SimpleCNN(
            num_classes=num_classes,
            conv_filters=config_dict.get('conv_filters', [32, 32, 32, 32]),
            fc_units=config_dict.get('fc_units', 128),
            dropout=config_dict.get('dropout', 0.5)
        )
    elif model_name == 'cnn_small':  # OLFED Student model (lightweight)
        return SimpleCNN_Small(
            num_classes=num_classes
        )
    elif model_name == 'cnn_medium':  # OLFED Student model (medium)
        return SimpleCNN_Medium(
            num_classes=num_classes
        )
    elif model_name == 'resnet18':
        return ResNet18NILM(
            num_classes=num_classes,
            pretrained=config_dict.get('pretrained', False)
        )
    
    elif model_name == 'mobilenet':
        return MobileNetV2NILM(
            num_classes=num_classes,
            width_mult=config_dict.get('width_mult', 1.0)
        )
    
    elif model_name == 'vgg16':
        return VGG16NILM(
            num_classes=num_classes,
            batch_norm=config_dict.get('batch_norm', True)
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test all models
    print("Testing model architectures...")
    
    num_classes = 11
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 192, 192)
    
    for model_name in ['cnn', 'resnet18', 'mobilenet', 'vgg16']:
        print(f"\n{model_name.upper()}:")
        model = create_model(model_name, num_classes)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {count_parameters(model):,}")
