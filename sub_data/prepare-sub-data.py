
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

# Load the model and tokenizer
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
caption_tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image, model, tokenizer):
    inputs = tokenizer(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

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

strings = []
y_noisy = []
y_real = []
failed_images = []
# encoded_images = []
c = 0

# Ensure the sub_data directory exists
os.makedirs('sub_data/images', exist_ok=True)

for img_path in images:
    print(c)
    c += 1
    if c > 10240:
        break
    try:
        # image_input = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        raw_img = Image.open(img_path)
        
        caption = generate_caption(raw_img, caption_model, caption_tokenizer)
        print("Caption:", caption)

        inputs = processor(images=raw_img, return_tensors="pt")
        with torch.no_grad():

            # Get CLIP embeddings
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings = clip.get_image_features(**inputs)
            embeddings = embeddings / torch.linalg.vector_norm(embeddings, dim=-1, keepdim=True)
            
            # Get the predicted score
            real_y = eval_model(embeddings).to(device)
            noisy_y = real_y + torch.randn_like(real_y,device=device) * 0.1
        
        # Save a copy of the image to the sub_data directory
        sub_data_path = os.path.join('sub_data/images', os.path.basename(img_path))
        shutil.copy(img_path, sub_data_path)
        
        # Write the image path and caption and score as 3 columns separated by comma
        strings.append(f"{sub_data_path},{caption},{real_y.item()}")    
        # x.append(embeddings.cpu().detach().numpy())
        y_noisy.append(noisy_y.cpu().detach().numpy())
        y_real.append(real_y.cpu().detach().numpy())

        
    except Exception as e:
        c -= 1
        print(f"Error processing image {img_path}: {e}")
        failed_images.append(img_path)
        continue


## saving figures

with open('sub_data/path_caption_score.csv', 'a', encoding='utf-8') as file:
    file.write("Img_path, Caption, Score\n")
    for string in strings:
        file.write(string + '\n')

# x = np.vstack(x)
# print(x.shape)
# np.save('sub_data/all_x_clip_embeddings.npy', x)

y_noisy = np.vstack(y_noisy)
print(y_noisy.shape)
np.save('sub_data/sub_y_noisy.npy', y_noisy)

y_real = np.vstack(y_real)
np.save('sub_data/sub_y_real.npy', y_real)
print(y_real.min(), y_real.max())