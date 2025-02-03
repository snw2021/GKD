#!/bin/bash
# example
# WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/kd.yaml

# same 
#CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/res56_res20.yaml
#CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/res110_res32.yaml
#CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/res32x4_res8x4.yaml
#CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/wrn40_2_wrn_16_2.yaml
#CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/wrn40_2_wrn_40_1.yaml
#CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/vgg13_vgg8.yaml

# different
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/res32x4_shuv1.yaml
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/wrn40_2_shuv1.yaml
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/vgg13_mv2.yaml
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/res50_mv2.yaml
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline python tools/train.py --cfg configs/cifar100/gkd/res32x4_shuv2.yaml

