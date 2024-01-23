
from sd_pipeline import GuidedSDPipeline
import torch
import numpy as np
from PIL import Image
import PIL
from typing import Callable, List, Optional, Union, Dict, Any
from dataset import CustomCIFAR10Dataset, CustomLatentDataset
from vae import encode
import os
from aesthetic_scorer import SinusoidalTimeMLP, MLPDiff
import wandb
import argparse



def parse():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", type=float, default=0.)
    parser.add_argument("--guidance", type=float, default=0.) 
    parser.add_argument("--prompt", type=str, default= "")
    parser.add_argument("--out_dir", type=str, default= "")
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=-1)

    args = parser.parse_args()
    return args


######### preparation ##########

args = parse()
device= args.device
save_file = True

## Image Seeds
if args.seed > 0:
    torch.manual_seed(args.seed)
    shape = (args.num_images//args.bs, args.bs , 4, 64, 64)
    init_latents = torch.randn(shape, device=device)
else:
    init_latents = None

if args.out_dir == "":
    args.out_dir = f'imgs/target_{args.target}_guidance{args.guidance}'
try:
    os.makedirs(args.out_dir)
except:
    pass

import prompts as prompts_file
eval_prompt_fn = getattr(prompts_file, 'eval_simple_animals')
eval_prompts, eval_prompt_metadata = zip(
    *[eval_prompt_fn() for _ in range(args.num_images)]
)

wandb.init(project="guided_dm", config={
    'target': args.target,
    'guidance': args.guidance, 
    'prompt': eval_prompts,
    'num_images': args.num_images
})


sd_model = GuidedSDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
sd_model.to(device)

sd_model.vae.requires_grad_(False)
sd_model.text_encoder.requires_grad_(False)
sd_model.unet.requires_grad_(False)

sd_model.vae.eval()
sd_model.text_encoder.eval()
sd_model.unet.eval()



reward_model = torch.load('model/reward_predictor_epoch_3.pth').to(device)
reward_model.eval()

sd_model.setup_reward_model(reward_model)
sd_model.set_target(args.target)
sd_model.set_guidance(args.guidance)

image = []
for i in range(args.num_images // args.bs):
    if init_latents is None:
        init_i = None
    else:
        init_i = init_latents[i]
    image_ = sd_model(eval_prompts, num_images_per_prompt=args.bs, latents=init_i).images # List of PIL.Image objects
    image.extend(image_)


###### evaluation and metric #####
from dataset import AVALatentDataset
pred_dataset= AVALatentDataset(image)
pred_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=20, shuffle=False, num_workers=8)


from dataset import AVACLIPDataset
gt_dataset= AVACLIPDataset(image)
gt_dataloader = torch.utils.data.DataLoader(gt_dataset, batch_size=20, shuffle=False, num_workers=8)

from importlib import resources
ASSETS_PATH = resources.files("assets")
eval_model = MLPDiff().to(device)
eval_model.requires_grad_(False)
eval_model.eval()
s = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=device)   # load the model you trained previously or the model available in this repo
eval_model.load_state_dict(s)

with torch.no_grad():
    total_reward_gt = []
    for inputs in gt_dataloader:
        inputs = inputs.to(device)
        gt_rewards = eval_model(inputs)
        
        print(gt_rewards, torch.mean(gt_rewards))
        
        total_reward_gt.append( gt_rewards.cpu().numpy() )

    total_reward_gt = np.concatenate(total_reward_gt, axis=None)

    wandb.log({"eval_reward_mean": np.mean(total_reward_gt) ,
               "eval_reward_std": np.std(total_reward_gt) })


with torch.no_grad():
    total_reward_pred= []
    for inputs in pred_dataloader:
        inputs = inputs.to(device)
        inputs = encode(inputs)
        
        timestep_list = torch.tensor([1]).to(inputs.device).repeat(len(inputs))
        pred_rewards = reward_model(inputs, timestep_list)
        
        print(pred_rewards, torch.mean(pred_rewards))
        
        total_reward_pred.append(pred_rewards.cpu().numpy())

    total_reward_pred = np.concatenate(total_reward_pred, axis=None)
    wandb.log({"reward_mean": np.mean(total_reward_pred) ,
               "reward_std": np.std(total_reward_pred) })


if save_file:
    for idx, im in enumerate(image):
        im.save(args.out_dir +'/'+ f'{idx}_gt_{total_reward_gt[idx]:.4f}_pred_{total_reward_pred[idx]:.4f}.png')
