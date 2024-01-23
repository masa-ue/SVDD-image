
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import torch.nn as nn
from vae import prepare_image

CIFAR10_Transform= transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def build_dataset(*,do_transform):
    # Load the CIFAR-10 dataset
    if do_transform:
        transform = CIFAR10_Transform
    else:
        transform = transforms.Compose([
        transforms.Resize(512),
    ])
    cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # use train-split

    return cifar10_dataset



# class CustomCIFAR10Dataset(torch.utils.data.Dataset):
#     def __init__(self, im_list):
#         self.data = im_list # List of PIL

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return CIFAR10_Transform(self.data[idx])


# class CustomLatentDataset(torch.utils.data.Dataset):
#     def __init__(self, im_list):
#         self.data = im_list # List of PIL

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return prepare_image(self.data[idx])[0] # prepare_image returns a tensor with shape [1, ...], the first dim is batch size.


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

        self.clip.requires_grad_(False)
        self.clip.eval()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        device="cuda"
        inputs = self.processor(images=self.data[idx], return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()} # Get CLIP embeddings
            self.clip.to(device)
            
            embeddings = self.clip.get_image_features(**inputs)
            embeddings = embeddings / torch.linalg.vector_norm(embeddings, dim=-1, keepdim=True)
        return embeddings[0]

if __name__ == "__main__":
    my_dataset = build_dataset(do_transform=False)
    im, label = my_dataset[0]
    print(type(im))
    print(im.height, im.width)
