
from sd_pipeline import GuidedSDPipeline
from diffusers import DDIMScheduler
import torch
import numpy as np
from PIL import Image
import PIL
from typing import Callable, List, Optional, Union, Dict, Any
from dataset import AVACLIPDataset, AVALatentDataset
from vae import encode
import os
from aesthetic_scorer import SinusoidalTimeMLP, MLPDiff
import wandb
import argparse
import datetime

import prompts as prompts_file
eval_prompt_fn = getattr(prompts_file, 'eval_simple_animals')



def parse():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", type=float, default=7.0)
    parser.add_argument("--guidance", type=float, default=100)
    parser.add_argument("--out_dir", type=str, default= "")
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--val_bs", type=int, default=4)
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

run_name = f"y_{args.target}_guidance_{args.guidance}"
unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
run_name = run_name + '_' + unique_id


if args.out_dir == "":
    args.out_dir = 'logs/' + run_name
try:
    os.makedirs(args.out_dir)
except:
    pass


wandb.init(project="RCGDM", name=run_name,config=args)


sd_model = GuidedSDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
sd_model.to(device)

# switch to DDIM scheduler
sd_model.scheduler = DDIMScheduler.from_config(sd_model.scheduler.config)
sd_model.scheduler.set_timesteps(50, device=device)

sd_model.vae.requires_grad_(False)
sd_model.text_encoder.requires_grad_(False)
sd_model.unet.requires_grad_(False)

sd_model.vae.eval()
sd_model.text_encoder.eval()
sd_model.unet.eval()


reward_model = torch.load('model/reward_predictor_epoch_3.pth').to(device)
reward_model.eval()
reward_model.requires_grad_(False)

sd_model.setup_reward_model(reward_model)
sd_model.set_target(args.target)
sd_model.set_guidance(args.guidance)


image = []
eval_prompt_list = []
KL_list = []

for i in range(args.num_images // args.bs):
    wandb.log(
        {"inner_iter": i}
    )
    if init_latents is None:
        init_i = None
    else:
        init_i = init_latents[i]
    eval_prompts, _ = zip(
        *[eval_prompt_fn() for _ in range(args.bs)]
    )
    eval_prompts = list(eval_prompts)
    eval_prompt_list.extend(eval_prompts)
    
    image_,kl_loss = sd_model(eval_prompts, num_images_per_prompt=1, eta=1.0, latents=init_i) # List of PIL.Image objects
    image.extend(image_)
    KL_list.append(kl_loss)

KL_entropy = torch.mean(torch.stack(KL_list))

assert len(image) == len(eval_prompt_list)

###### evaluation and metric #####
gt_dataset= AVACLIPDataset(image)
gt_dataloader = torch.utils.data.DataLoader(gt_dataset, batch_size=args.val_bs, shuffle=False)

from importlib import resources
ASSETS_PATH = resources.files("assets")
eval_model = MLPDiff().to(device)
eval_model.requires_grad_(False)
eval_model.eval()
s = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=device)   # load the model you trained previously or the model available in this repo
eval_model.load_state_dict(s)

with torch.no_grad():
    total_reward_gt = []
    total_reward_pred= []
    for inputs in gt_dataloader:
        inputs = inputs.to(device)
        
        ### Obtaining ground truth reward
        gt_rewards = eval_model(inputs)
        # print("eval_rewards_mean: ", torch.mean(gt_rewards))
        total_reward_gt.append(gt_rewards.cpu().numpy())
        
        ### Obtaining predicted reward
        timestep_list = torch.tensor([1]).to(inputs.device).repeat(len(inputs))
        pred_rewards = reward_model(inputs, timestep_list)
        
        # print("rewards_mean: ", torch.mean(pred_rewards))
        total_reward_pred.append(pred_rewards.cpu().numpy())

    total_reward_gt = np.concatenate(total_reward_gt, axis=None)
    total_reward_pred = np.concatenate(total_reward_pred, axis=None)

    print("eval_reward_mean: ", np.mean(total_reward_gt) )
    print("reward_mean: ", np.mean(total_reward_pred) )
    print("KL-entropy: ", KL_entropy)
    
    wandb.log({"eval_reward_mean": np.mean(total_reward_gt) ,
               "eval_reward_std": np.std(total_reward_gt) })
    wandb.log({"reward_mean": np.mean(total_reward_pred) ,
               "reward_std": np.std(total_reward_pred) })
    wandb.log({"KL-entropy": KL_entropy })

if save_file:
    images = []
    for idx, im in enumerate(image):
        im.save(args.out_dir +'/'+ f'{idx}_gt_{total_reward_gt[idx]:.4f}_pred_{total_reward_pred[idx]:.4f}.png')
        
        pil = im.resize((256, 256))
        prompt = eval_prompt_list[idx]
        reward = total_reward_gt[idx]
        images.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))

    wandb.log(
        {"images": images}
    )