
from torch.utils.data import Dataset, DataLoader
import clip
import torch,torchvision
from PIL import Image, ImageFile
import torch.nn as nn
import shutil
import numpy as np
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import sys
import json
sys.path.append(os.getcwd())

import contextlib
import io

from vae import prepare_image, encode, decode_latents

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

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            path = os.path.join(folder, filename)
            images.append(path)
    return images

# Set up the device for GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load images from the folder
folder = './sub_data/train'
images = load_images_from_folder(folder)
print("Number of images:", len(images))


y_real = []
failed_images = []
c = 0

data_list = []

target_size = 512
for img_path in images:
    print(c)
    c += 1
    if c > 10240:
        break
    try:
        # image_input = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        raw_img = Image.open(img_path)
        
        
        # resized_img = torchvision.transforms.Resize((512, 512))(raw_img)
        # img_tensor = prepare_image(resized_img)
        
        raw_img = raw_img.convert('RGB')
        transform = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                    ])
        img_tensor = transform(raw_img)
                
        real_y = jpeg_compressibility(img_tensor.unsqueeze(0))

        y_real.append(real_y)
        
    except Exception as e:
        c -= 1
        print(f"Error processing image {img_path}: {e}")
        failed_images.append(img_path)
        continue

y_real = np.vstack(y_real)
print(y_real.shape)
np.save('sub_data/sub_y_real-compressibility.npy', y_real)
print(y_real.min(), y_real.max())