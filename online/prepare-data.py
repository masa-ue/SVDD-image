import torch
from PIL import Image
import sys
import os
import copy
import gc
cwd = os.getcwd()
sys.path.append(cwd)

from aesthetic_scorer import MLPDiff, AestheticScorerDiff
from transformers import CLIPModel, CLIPProcessor
import torchvision

from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
import matplotlib.pyplot as plt
import sys

from vae import sd_model

    
from importlib import resources
ASSETS_PATH = resources.files("assets")

encoded_imgs = torch.load(f"./model/pre_generated_latents-15360.pth")
print(encoded_imgs.shape)

latents_dataset = torch.utils.data.TensorDataset(encoded_imgs)
train_loader = torch.utils.data.DataLoader(latents_dataset, batch_size=16, shuffle=False)

device = 'cuda'
eval_model = MLPDiff().to(device)

eval_model.load_state_dict(torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth")))
eval_model.requires_grad_(False)
eval_model.eval()


normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
resize = torchvision.transforms.Resize(224, antialias=False)
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip.eval()
clip.to(device)

noisy_y_list = []
real_y_list = []

for i, (inputs,) in enumerate(tqdm(train_loader, desc="Evaluating Progress")):
    with torch.no_grad():
        print(i)
        inputs = inputs.to(device)
        
        im_pix_un = sd_model.vae.decode(inputs.to(sd_model.vae.dtype) / 0.18215).sample
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = resize(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        embed = clip.get_image_features(pixel_values=im_pix)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        assert embed.shape == (inputs.shape[0], 768)
        assert embed.requires_grad == False

    real_y = eval_model(embed)
    noisy_y = real_y + torch.randn_like(real_y,device=device) * 0.1
    
    real_y_list.append(real_y)
    noisy_y_list.append(noisy_y)
    
noisy_y = torch.cat(noisy_y_list, dim=0)
print(noisy_y.shape)
torch.save(noisy_y, "./model/pre_generated_scores-15360.pth")

real_y = torch.cat(real_y_list, dim=0)
real_y = real_y.cpu().detach().numpy()
print(real_y.shape, real_y.min(), real_y.max())
plt.hist(real_y[:,0], bins='auto', edgecolor='black')

# Add titles and labels
plt.title(f'Histogram of SD Generated Dataset ({len(real_y)} images)')
plt.xlabel('Aesthetic Visual Analysis (AVA) Score (1-10)')
plt.ylabel('Frequency')
plt.savefig("./model/hist_v1.png")