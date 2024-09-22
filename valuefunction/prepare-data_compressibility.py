import torch
from pathlib import Path
import numpy as np
import sys
import os
from PIL import Image, UnidentifiedImageError
import io
import contextlib

cwd = os.getcwd()
sys.path.append(cwd)

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import warnings

def jpeg_compressibility(images):
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    pil_images = [Image.fromarray(image) for image in images]
    # buffers = [io.BytesIO() for _ in images]
    # for image, buffer in zip(images, buffers):
    #     image.save(buffer, format="JPEG", quality=95)
    # sizes = [buffer.tell() / 1000 for buffer in buffers]
    sizes = []
    with contextlib.ExitStack() as stack:
        buffers = [stack.enter_context(io.BytesIO()) for _ in pil_images]
        for image, buffer in zip(pil_images, buffers):
            image.save(buffer, format="JPEG", quality=95)
            sizes.append(buffer.tell() / 1000)  # Size in kilobytes
    
    return -np.array(sizes)

class CompressibilityDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        self.problematic_images = []
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                    ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with Image.open(img_path) as img:
                    image = img.convert('RGB')

                # Check for warnings related to EXIF data
                for warning in w:
                    if "Corrupt EXIF data" in str(warning.message):
                        self.problematic_images.append(img_path)
                        print(f"Warning for image {img_path}: {warning.message}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
        
        if self.transform:
            image_tensor = self.transform(image)
        
        compressibility = jpeg_compressibility(image_tensor.unsqueeze(0))
        
        return image_tensor, torch.tensor(compressibility[0], dtype=torch.float32)

# image_dir = './sub_data/train'
image_dir = './images'

def draw_hist_v1(image_dir):
    dataset = CompressibilityDataset(image_dir=image_dir)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8)
    
    y = list()
    for _, (image_tensors,targets) in enumerate(tqdm(train_loader)):        
        y.append(targets.cpu().detach().numpy())
    
    y = np.concatenate(y, axis=0)
    
    print(y.shape, y.min(), y.max())
    
    print(y.shape)
    np.save('./data/y_real-compressibility.npy', y)
    
    # Calculating quantiles
    q25 = np.percentile(y, 25)  # 25th percentile
    median = np.percentile(y, 50)  # 50th percentile (Median)
    q75 = np.percentile(y, 75)  # 75th percentile
    
    # Printing quantiles
    print("25% Quantile:", q25) # -110
    print("Median:", median)    # -84.5
    print("75% Quantile:", q75) # -63.8
    
    plt.figure()
    counts, bins, patches = plt.hist(y, bins='auto', edgecolor='black')
    plt.title(f'Histogram of AVA ({len(y)} images)')
    plt.xlabel('Compressibility scores for AVA dataset')
    plt.ylabel('Frequency')
    plt.savefig("./data/Comp_v2_hist.png")
    plt.close()

draw_hist_v1(image_dir)
