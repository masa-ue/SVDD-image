import torch
import numpy as np
import sys
import os

cwd = os.getcwd()
sys.path.append(cwd)

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


def draw_hist(): # draw histogram of AVA dataset
    real_y = np.load("./new_data/all_y_real.npy")   # 2.7267294 7.0839090
    print(real_y.shape, real_y.min(), real_y.max())
    p95 = np.percentile(real_y[:,0], 95)
    
    plt.hist(real_y[:, 0], bins='auto', edgecolor='black')
    
    # Plot the 90th percentile line
    plt.axvline(p95, color='red', linestyle='dashed', linewidth=2)

    # Annotate the 90th percentile line
    plt.text(p95, plt.ylim()[1]*0.9, f'95th percentile: {p95:.2f}', color='red')

    # Add titles and labels
    plt.title(f'Histogram of AVA Dataset ({len(real_y)} images)')
    plt.xlabel('Aesthetic Visual Analysis (AVA) Score (1-10)')
    plt.ylabel('Frequency')
    plt.savefig("./new_data/AVA_hist_v2.png")
    plt.show()

draw_hist()

# x = torch.tensor(np.load("./new_data/all_x_clip_embeddings.npy"))
# print(x.shape)
# print(x.dtype)
# print(torch.norm(x, dim=1))

# latents = torch.tensor(np.load("./new_data/all_vae_latents.npy"))
# print(latents.shape)

