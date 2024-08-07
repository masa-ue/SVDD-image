
from sd_pipeline import DPS_multitask_SDPipeline
from diffusers import DDIMScheduler
import torch
import numpy as np
from PIL import Image
import PIL
from typing import Callable, List, Optional, Union, Dict, Any
from dataset import AVACompressibilityDataset, AVACLIPDataset
from vae import encode
import os
from aesthetic_scorer import SinusoidalTimeMLP, MLPDiff
import wandb
import argparse
from tqdm import tqdm
import datetime
from compressibility_scorer import condition_CompressibilityScorerDiff, jpeg_compressibility, classify_compressibility_scores_4class, classify_compressibility_scores
from aesthetic_scorer import classify_aesthetic_scores_easy, condition_AestheticScorerDiff, MLPDiff

from diffusers_patch.utils import compute_classification_metrics

import prompts as prompts_file
eval_prompt_fn = getattr(prompts_file, 'eval_aesthetic_animals')



def parse():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--comp_target", type=int, default=0)
    parser.add_argument("--aesthetic_target", type=int, default=0)
    
    parser.add_argument("--comp_weight", type=float, default=1.0)
    parser.add_argument("--aesthetic_weight", type=float, default=1.0)
    
    parser.add_argument("--guidance", type=float, default=0)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--val_bs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

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

run_name = f"Class=({args.aesthetic_target},{args.comp_target})_gamma={args.guidance}_{args.num_images}_weights=({args.aesthetic_weight},{args.comp_weight})"
unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
run_name = run_name + '_' + unique_id


if args.out_dir == "":
    args.out_dir = 'logs/' + run_name
try:
    os.makedirs(args.out_dir)
except:
    pass


wandb.init(project=f"DPS-multitask-({args.aesthetic_target},{args.comp_target})", name=run_name,config=args)


sd_model = DPS_multitask_SDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
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

comp_scorer = condition_CompressibilityScorerDiff(dtype=torch.float32).to(device)
comp_scorer.requires_grad_(False)
comp_scorer.eval()

aesthetic_scorer = condition_AestheticScorerDiff(dtype=torch.float32).to(device)
aesthetic_scorer.requires_grad_(False)
aesthetic_scorer.eval()

sd_model.setup_comp_scorer(comp_scorer)
sd_model.set_comp_target(args.comp_target)
sd_model.set_guidance(args.guidance)

sd_model.setup_aesthetic_scorer(aesthetic_scorer)
sd_model.set_aesthetic_target(args.aesthetic_target)

sd_model.set_aesthetic_weight(args.aesthetic_weight)
sd_model.set_comp_weight(args.comp_weight)


image = []
eval_prompt_list = []
KL_list = []

for i in tqdm(range(args.num_images // args.bs), desc="Generating Images"):
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
gt_dataset= AVACompressibilityDataset(image)
gt_dataloader = torch.utils.data.DataLoader(gt_dataset, batch_size=args.val_bs, shuffle=False)

wandb.log({"KL-entropy": KL_entropy })

with torch.no_grad():
    total_class_labels_comp = []
    total_predicted_classes_comp = []
    eval_rewards_comp = []
    
    for inputs in gt_dataloader:
        inputs = inputs.to(device)
        eval_labels = torch.tensor([args.comp_target]*len(inputs)).to(device)

        jpeg_compressibility_scores = jpeg_compressibility(inputs)
        comp_scores = torch.tensor(jpeg_compressibility_scores, dtype=inputs.dtype, device=inputs.device)
        predicted_classes = classify_compressibility_scores(comp_scores)
        
        eval_rewards_comp.extend(comp_scores.tolist())
        total_class_labels_comp.extend(eval_labels.tolist())
        total_predicted_classes_comp.extend(predicted_classes.tolist())

    total_class_labels_comp = torch.tensor(total_class_labels_comp)
    total_predicted_classes_comp = torch.tensor(total_predicted_classes_comp)
    eval_rewards_comp = torch.tensor(eval_rewards_comp)
    
    metrics = compute_classification_metrics(total_predicted_classes_comp, total_class_labels_comp)

    print(f"eval_class_{args.comp_target}_rewards_mean_compressibility", torch.mean(eval_rewards_comp))
    print("eval_accuracy_compressibility",metrics['accuracy'])
    print("eval_macro_F1_compressibility", metrics['macro_F1'])
    
    wandb.log({
        f"eval_class_{args.comp_target}_rewards_mean_compressibility": torch.mean(eval_rewards_comp),
        "eval_accuracy_compressibility":metrics['accuracy'],
        "eval_macro_F1_compressibility": metrics['macro_F1'],
    })

from importlib import resources
ASSETS_PATH = resources.files("assets")
eval_model = MLPDiff().to(device)
eval_model.requires_grad_(False)
eval_model.eval()
s = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=device)   # load the model you trained previously or the model available in this repo
eval_model.load_state_dict(s)

gt_dataset= AVACLIPDataset(image)
gt_dataloader = torch.utils.data.DataLoader(gt_dataset, batch_size=args.val_bs, shuffle=False)

wandb.log({"KL-entropy": KL_entropy })

with torch.no_grad():
    total_class_labels_aesthetic = []
    total_predicted_classes_aesthetic = []
    eval_rewards_aesthetic = []
    
    for inputs in gt_dataloader:
        inputs = inputs.to(device)
        eval_labels = torch.tensor([args.aesthetic_target]*len(inputs)).to(device)

        aesthetic_scores = eval_model(inputs)
        predicted_classes = classify_aesthetic_scores_easy(aesthetic_scores)
        
        eval_rewards_aesthetic.extend(aesthetic_scores.tolist())
        total_class_labels_aesthetic.extend(eval_labels.tolist())
        total_predicted_classes_aesthetic.extend(predicted_classes.tolist())

    total_class_labels_aesthetic = torch.tensor(total_class_labels_aesthetic)
    total_predicted_classes_aesthetic = torch.tensor(total_predicted_classes_aesthetic)
    eval_rewards_aesthetic = torch.tensor(eval_rewards_aesthetic)
    
    metrics = compute_classification_metrics(total_predicted_classes_aesthetic, total_class_labels_aesthetic)

    print(f"eval_class_{args.aesthetic_target}_rewards_mean_aesthetic", torch.mean(eval_rewards_aesthetic))
    print("eval_accuracy_aesthetic",metrics['accuracy'])
    print("eval_macro_F1_aesthetic", metrics['macro_F1'])
    
    wandb.log({
        f"eval_class_{args.aesthetic_target}_rewards_mean_aesthetic": torch.mean(eval_rewards_aesthetic),
        "eval_accuracy_aesthetic":metrics['accuracy'],
        "eval_macro_F1_aesthetic": metrics['macro_F1'],
    })

if save_file:
    images = []
    log_dir = os.path.join(args.out_dir, "eval_vis")
    os.makedirs(log_dir, exist_ok=True)
    
    # Function to save array to a text file with commas
    def save_array_to_text_file(array, file_path):
        with open(file_path, 'w') as file:
            array_str = ','.join(map(str, array.tolist()))
            file.write(array_str + ',')

    # Save the arrays to text files
    save_array_to_text_file(eval_rewards_comp, f"{args.out_dir}/eval_rewards_comp.txt")
    save_array_to_text_file(eval_rewards_aesthetic, f"{args.out_dir}/eval_rewards_aesthetic.txt")
    
    save_array_to_text_file(total_class_labels_comp, f"{args.out_dir}/total_class_labels_comp.txt")
    save_array_to_text_file(total_predicted_classes_comp, f"{args.out_dir}/total_predicted_classes_comp.txt")
    
    save_array_to_text_file(total_class_labels_aesthetic, f"{args.out_dir}/total_class_labels_aesthetic.txt")
    save_array_to_text_file(total_predicted_classes_aesthetic, f"{args.out_dir}/total_predicted_classes_aesthetic.txt")

    print("Arrays have been saved to text files.")
    
    for idx, im in enumerate(image):
        # im.save(args.out_dir +'/'+ f'{idx}_gt_{total_reward_gt[idx]:.4f}_pred_{total_reward_pred[idx]:.4f}.png')
        prompt = eval_prompt_list[idx]
        
        label_comp = total_class_labels_comp[idx]
        label_aesthetic = total_class_labels_aesthetic[idx]
        # reward = eval_rewards[idx]
        predicted_class_comp = total_predicted_classes_comp[idx]
        predicted_class_aesthetic = total_predicted_classes_aesthetic[idx]
        
        im.save(f"{log_dir}/{idx:03d}_{prompt}_result=({predicted_class_aesthetic}, {predicted_class_comp})_condition=({label_aesthetic}, {label_comp}).png")
        
        pil = im.resize((256, 256))

        images.append(wandb.Image(pil, caption=f"{prompt:.25} | result=({predicted_class_aesthetic}, {predicted_class_comp}) | condition=({label_aesthetic}, {label_comp})"))

    wandb.log(
        {"images": images}
    )