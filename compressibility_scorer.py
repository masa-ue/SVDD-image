# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

import os,sys
cwd = os.getcwd()
sys.path.append(cwd)

from importlib import resources
import torch
import torch.nn as nn
import numpy as np
import math
import random
import torch.nn.functional as F
from transformers import CLIPModel
from PIL import Image
from torch.utils.checkpoint import checkpoint

import contextlib
import io
from PIL import Image

from diffusers_patch.utils import TemperatureScaler

class JPEG_class:
    def __init__(self):
        pass 
    def jpeg_compressibility(self,images):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        pil_images = [Image.fromarray(image) for image in images]

        sizes = []
        with contextlib.ExitStack() as stack:
            buffers = [stack.enter_context(io.BytesIO()) for _ in pil_images]
            for image, buffer in zip(pil_images, buffers):
                image.save(buffer, format="JPEG", quality=95)
                sizes.append(buffer.tell() / 1000)  # Size in kilobytes
        
        return -np.array(sizes)
    
def jpeg_compressibility(images):
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    pil_images = [Image.fromarray(image) for image in images]

    sizes = []
    with contextlib.ExitStack() as stack:
        buffers = [stack.enter_context(io.BytesIO()) for _ in pil_images]
        for image, buffer in zip(pil_images, buffers):
            image.save(buffer, format="JPEG", quality=95)
            sizes.append(buffer.tell() / 1000)  # Size in kilobytes
    
    return -np.array(sizes)

def classify_compressibility_scores(y):
    # Applying thresholds to map scores to classes
    class_labels = torch.zeros_like(y, dtype=torch.long)  # Ensure it's integer type for class labels
    class_labels[y >= - 70.0] = 1
    class_labels[y < -70.0] = 0
    if class_labels.dim() > 1:
        return class_labels.squeeze(1)
    return class_labels

def classify_compressibility_scores_4class(y):
    # Applying thresholds to map scores to classes
    class_labels = torch.zeros_like(y, dtype=torch.long)  # Ensure it's integer type for class labels
    class_labels[y >= - 60.0] = 3
    class_labels[(y < -60.0) & (y >= -85.0)] = 2
    class_labels[(y < -85.0) & (y >= -110.0)] = 1
    class_labels[y < -110.0] = 0
    if class_labels.dim() > 1:
        return class_labels.squeeze(1)
    return class_labels

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ThreeLayerConvNet(nn.Module):
    def __init__(self, num_channels, num_classes=1, dtype=torch.float32):
        super(ThreeLayerConvNet, self).__init__()
        
        self.dtype = dtype
        self.layer1 = ResidualBlock(num_channels, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.layer1(x.to(self.dtype))
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class SinusoidalTimeConvNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=1, time_encoding_dim=64, dtype=torch.float32):
        super(SinusoidalTimeConvNet, self).__init__()
        
        self.dtype = dtype
        self.time_encoding_dim = time_encoding_dim

        # Standard convolutional layers
        self.layer1 = ResidualBlock(num_channels, 64, stride=1)
        self.layer2 = ResidualBlock(64 + time_encoding_dim, 128, stride=2)  # Concatenating time embedding here
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def sinusoidal_encoding(self, timesteps, height, width):
        # Normalize timesteps to be in the range [0, 1]
        timesteps = timesteps.float() / 1000.0  # Assuming timesteps are provided as integers

        # Generate a series of frequencies for the sinusoidal embeddings
        frequencies = torch.exp(torch.arange(0, self.time_encoding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / self.time_encoding_dim))
        frequencies = frequencies.to(timesteps.device)

        # Apply the frequencies to the timesteps
        arguments = timesteps[:, None] * frequencies[None, :]
        encoding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)

        # Reshape the time embedding to match the spatial dimensions (height, width)
        encoding = encoding[:, :, None, None].repeat(1, 1, height, width)  # Repeat for spatial dimensions
        return encoding

    def forward(self, x, timesteps):
        batch_size, channels, height, width = x.size()

        # Pass through the first convolutional layer
        out = self.layer1(x.to(self.dtype))

        # Generate sinusoidal embeddings for the timesteps and expand to match the feature map dimensions
        timestep_embed = self.sinusoidal_encoding(timesteps, out.size(2), out.size(3))

        # Concatenate the time embedding with the output of the first layer
        combined_input = torch.cat([out, timestep_embed], dim=1)

        # Continue with the remaining convolutional layers
        out = self.layer2(combined_input)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # Flatten the feature map
        out = self.fc(out)
        return out

class CompressibilityScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def __call__(self, images):
        jpeg_compressibility_scores = jpeg_compressibility(images)
        return torch.tensor(jpeg_compressibility_scores, dtype=images.dtype, device=images.device), images

class CompressibilityScorer_modified(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def __call__(self, images):
        jpeg_compressibility_scores = jpeg_compressibility(images)
        return jpeg_compressibility_scores


class condition_CompressibilityScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
        
        # state_dict = torch.load('comp_model/CNN_5class_v1_64_final_calibrated.pth')
        state_dict = torch.load('comp_model/CNN_3class_v3_final_calibrated.pth')
    
        self.scaler = TemperatureScaler()
        self.scaler.load_state_dict(state_dict['scaler'])
        
        self.model = ThreeLayerConvNet(num_channels=3, num_classes=3)
        self.model.load_state_dict(state_dict['model_state_dict'])
        
        self.eval()

    def __call__(self, images):
        logits = self.model(images)
        calibrated_logits = self.scaler(logits)
        probabilities = F.softmax(calibrated_logits, dim=1)
        
        return probabilities, images

class CompressibilityScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
        
        # state_dict = torch.load('comp_model/lr=1e-2_2024.06.27_04.52.36_38.pth', weights_only=True)
        # self.model = ThreeLayerConvNet(num_channels=3, num_classes=1)
        # self.model.load_state_dict(state_dict)
        self.model = torch.load('comp_model/reward_predictor_epoch_199.pth')
        
        self.eval()

    def __call__(self, images, timesteps):
        predictions = self.model(images, timesteps).squeeze(1)  # images: (B, 3, 512, 512), timesteps: (N,)
        
        return predictions, images

if __name__ == "__main__":
    scorer = condition_CompressibilityScorerDiff(dtype=torch.float32).cuda()
    scorer.requires_grad_(False)
    scorer.eval()
    
    for param in scorer.model.parameters():
        assert not param.requires_grad, "Model parameters should not require gradients"