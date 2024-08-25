
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset
from vae import prepare_image

class AVALatentDataset(torch.utils.data.Dataset):
    def __init__(self, im_list):
        self.data = im_list # List of PIL

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transformed_img = torchvision.transforms.Resize((512,512))(self.data[idx])
        return prepare_image(transformed_img)[0] # prepare_image returns a tensor with shape [1, ...], the first dim is batch size.

class AVACLIPDataset(torch.utils.data.Dataset):
    def __init__(self, im_list):
        self.data = im_list # List of PIL
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        device="cuda"
        self.clip.to(device)

        self.clip.requires_grad_(False)
        self.clip.eval()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        device="cuda"
        inputs = self.processor(images=self.data[idx], return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()} # Get CLIP embeddings
            embeddings = self.clip.get_image_features(**inputs)
            embeddings = embeddings / torch.linalg.vector_norm(embeddings, dim=-1, keepdim=True)
        return embeddings[0]

class AVACompressibilityDataset(torch.utils.data.Dataset):
    def __init__(self, im_list):
        self.data = im_list # List of PIL

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transform = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                    ])
        transformed_img = transform(self.data[idx])
        return transformed_img

class ImageDataset(Dataset):
    def __init__(self, directory):
        self.data = self.load_images_from_directory(directory)

    def load_images_from_directory(self, directory):
        image_list = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(root, file)
                    image = Image.open(img_path).convert("RGB")  # Ensure all images are RGB
                    image_list.append(image)
        return image_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transform = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
                        
                    ])
        transformed_img = transform(self.data[idx])
        return transformed_img
