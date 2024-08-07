## DPS baseline with continuous regressors

### 1. compressibility

for compressibility, run the following. From my (preliminary) observation, I feel guidance here can be from 1~3. Even guidance strength=3 might be too strong as many generations loss fidelity. The rewards are saved to `scores.npy` and `eval_rewards.txt` and can be directly loaded.  
(I feel 1.0, 2.0 seem to be good.)

Note that by default, in inference the batch size is `2`, which would take `~27` Gb CUDA memory.

```
CUDA_VISIBLE_DEVICES=4 python inference_DPS_regressor.py --reward compressibility --guidance 1 --num_images=128
```

### 2. aesthetic scores

for aesthetic scores, run the following. From my (preliminary) observation, I feel guidance here can be from 0.5~2.0. You may try some values in this range, and as you may see, this DPS guidance does not make a big difference compared to pre-trained model... Guidance strength=2.5 may even be too strong which leads to decreased fidelity.  
(I feel 1.0, 1.5 seem to be good.)

```
CUDA_VISIBLE_DEVICES=4 python inference_DPS_regressor.py --reward aesthetic --guidance 0.1 --num_images=128
```

#### Current results

For aesthetic scores, I have added generated samples from the pre-trained model and their statistics in `making-plots/Eval_Pretrained-512_2024.05.17_13.52.31`.

### Requirements

Actually I recommend directly using the `alignprop` env rather than following the command below

```
pip install -r requirements.txt
```
