import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from aesthetic_scorer import SinusoidalTimeMLP

from accelerate import Accelerator
from accelerate.logging import get_logger 

from vae import sd_model
import wandb
logger = get_logger(__name__)

### Configs #### 

lr = 0.001
num_data = 100000
num_epochs = 10
batch_size = 16 # Batch size for training
# stable diffusion hyperparameters.
latent_dim = 4  
num_inference_steps = 50

config={
    'lr': lr,
    'num_data':num_data,
    'num_epochs':num_epochs,
    'batch_size':batch_size,
    'latent_dim':latent_dim,
}

wandb.init(project="SVDD-ValueFunction-aesthetic", config=config)
logger.info(f"\n{config}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = SinusoidalTimeMLP().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Prepare the training dataset with the encoded images and noisy outputs
targets = torch.tensor(np.load('new_data/all_y_noisy.npy'), dtype=torch.float32)
encoded_imgs = torch.tensor(np.load('new_data/all_vae_latents.npy'), dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(encoded_imgs[:num_data], targets[:num_data])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


sd_model.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = sd_model.scheduler.timesteps

normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
resize = torchvision.transforms.Resize(224, antialias=False)
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip.eval()
clip.to(device)

for epoch in tqdm(range(num_epochs), desc="Epoch"):
    model.train()
    epoch_loss = 0.0
    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training Progress")):
        
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)

            # Add random noise to the latent.
            random_sampled_timesteps = timesteps[torch.randint(low=0, high=len(timesteps), size=(inputs.shape[0],), device = device)]
            random_noise =  torch.randn_like(inputs, device = device)
            latent = sd_model.scheduler.add_noise(original_samples = inputs, noise = random_noise, timesteps = random_sampled_timesteps)

            # Forward pass
            im_pix_un = sd_model.vae.decode(latent.to(sd_model.vae.dtype) / 0.18215).sample
            im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
            im_pix = resize(im_pix)
            im_pix = normalize(im_pix).to(im_pix_un.dtype)
            embed = clip.get_image_features(pixel_values=im_pix)
            embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
            assert embed.shape == (inputs.shape[0], 768)
            assert embed.requires_grad == False
        
        outputs = model(embed, random_sampled_timesteps)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        wandb.log({"iter_loss": loss.item(), "epoch": epoch, "iter": i})
        logger.info(f"iter_loss: {loss.item()}")

        epoch_loss += loss.item()
        
        if i % 2000 == 0 and i != 0:
            torch.save(model, f'aes_model/reward_predictor_epoch_{epoch}_iter_{i}.pth')

    wandb.log({"epoch_loss": epoch_loss/(i+1)})
    logger.info(f"epoch_loss: {epoch_loss/(i+1):.4f}")
    
    torch.save(model, f'aes_model/reward_predictor_epoch_{epoch}.pth')

