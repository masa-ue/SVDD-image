import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from compressibility_scorer import SinusoidalTimeConvNet
from dataset import ImageDataset
from vae import prepare_image,  encode, sd_model

from accelerate import Accelerator
from accelerate.logging import get_logger 

from vae import sd_model
import wandb
logger = get_logger(__name__)

### Configs #### 

lr = 0.0001
num_data = 100000
num_epochs = 300
batch_size = 512 # Batch size for training
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

wandb.init(project="SVDD-ValueFunction-compressibility", config=config)
logger.info(f"\n{config}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

convnet = SinusoidalTimeConvNet(latent_dim, num_classes=1).to(device)

# my_dataset = ImageDataset('./images')
# encoded_images = []
# with torch.no_grad():
#     for data in tqdm(my_dataset, desc="Processing images"):
#         # data = prepare_image(data)
#         data = data.to(device)
#         encoded_img = encode(data.unsqueeze(0))
#         encoded_images.append(encoded_img)

# encoded_images = torch.cat(encoded_images, dim = 0).cpu()
# torch.save(encoded_images, f"./data/encoded_images_{len(encoded_images)}.pth")

encoded_images = torch.load("./data/encoded_images_84600.pth")

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(convnet.parameters(), lr=lr)

# Prepare the training dataset with the encoded images and noisy outputs
comp_labels = np.load('./data/y_real-compressibility.npy')
my_targets = torch.tensor(comp_labels, dtype=torch.float32).view(-1, 1)

train_dataset = torch.utils.data.TensorDataset(encoded_images[:num_data], my_targets[:num_data])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# Train the model

# noisy input
sd_model.scheduler.set_timesteps(num_inference_steps, device=device) # PNDM scheduler
timesteps = sd_model.scheduler.timesteps


for epoch in tqdm(range(num_epochs), desc="Epoch"):
    convnet.train()
    epoch_loss = 0.0
    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training Progress")):
        inputs, targets = inputs.to(device), targets.to(device)

        # Add random noise to the latent.
        random_sampled_timesteps = timesteps[torch.randint(low=0, high=len(timesteps), size=(inputs.shape[0],), device = device)]
        random_noise =  torch.randn_like(inputs, device = device)
        inputs = sd_model.scheduler.add_noise(original_samples = inputs, noise = random_noise, timesteps = random_sampled_timesteps)

        # Forward pass
        outputs = convnet(inputs, random_sampled_timesteps)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        wandb.log({"iter_loss": loss.item(), "epoch": epoch, "iter": i})
        logger.info(f"iter_loss: {loss.item()}")

        epoch_loss += loss.item()

    wandb.log({"epoch_loss": epoch_loss/(i+1)})
    logger.info(f"epoch_loss: {epoch_loss/(i+1):.4f}")

    torch.save(convnet, f'./comp_model/reward_predictor_epoch_{epoch}.pth')