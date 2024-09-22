
from torch.utils.data import Dataset, DataLoader
import clip
import torch,torchvision
from PIL import Image, ImageFile
import torch.nn as nn
import numpy as np
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import os
import sys
sys.path.append(os.getcwd())

from importlib import resources
ASSETS_PATH = resources.files("assets")

from vae import prepare_image, encode, decode_latents

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)
    
    def forward_up_to_second_last(self, embed):
        # Process the input through all layers except the last one
        for layer in list(self.layers)[:-1]:
            embed = layer(embed)
        return embed

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            # print(filename)
            path = os.path.join(folder, filename)
            images.append(path)
    return images

# Set up the device for GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load images from the folder
folder = './images'
images = load_images_from_folder(folder)
print("Number of images:", len(images))

with open("data/images.txt", "w") as file:
    for item in images:
        file.write(item + "\n")


### Generate CLIP Embeddings
# model, preprocess = clip.load("ViT-L/14", device=device)
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

clip.to(device)
clip.requires_grad_(False)
clip.eval()

eval_model = MLPDiff().to(device)
eval_model.requires_grad_(False)
eval_model.eval()
s = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=device)   # load the model you trained previously or the model available in this repo
eval_model.load_state_dict(s)

x = []
y_noisy = []
y_real = []
failed_images = []
encoded_images = []
c = 0

for img_path in images:
    print(c)
    c += 1
    try:
        # image_input = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        raw_img = Image.open(img_path)
        
        inputs = processor(images=raw_img, return_tensors="pt")
        with torch.no_grad():

            # Get CLIP embeddings
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings = clip.get_image_features(**inputs)
            embeddings = embeddings / torch.linalg.vector_norm(embeddings, dim=-1, keepdim=True)
            
            # Get the predicted score
            real_y = eval_model(embeddings).to(device)
            noisy_y = real_y + torch.randn_like(real_y,device=device) * 0.1

            # Get the encoded image (VAE latents)
            transformed_img = torchvision.transforms.Resize((512,512))(raw_img)
            data = prepare_image(transformed_img)
            encoded_img = encode(data.to(device))
        
            
        x.append(embeddings.cpu().detach().numpy())
        y_noisy.append(noisy_y.cpu().detach().numpy())
        y_real.append(real_y.cpu().detach().numpy())
        encoded_images.append(encoded_img.cpu().detach().numpy())
        
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        failed_images.append(img_path)
        continue

with open("new_data/all_failures.txt", "w") as file:
    for item in failed_images:
        file.write(item + "\n")

x = np.vstack(x)
print(x.shape)
np.save('new_data/all_x_clip_embeddings.npy', x)

y_noisy = np.vstack(y_noisy)
print(y_noisy.shape)
np.save('new_data/all_y_noisy.npy', y_noisy)

y_real = np.vstack(y_real)
np.save('new_data/all_y_real.npy', y_real)
print(y_real.min(), y_real.max())

encoded_images = np.vstack(encoded_images)
print(encoded_images.shape)
np.save('new_data/all_vae_latents.npy', encoded_images)