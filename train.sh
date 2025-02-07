#!/bin/bash

eval "$(conda shell.bash hook)"
# export CUDA_VISIBLE_DEVICES=1
conda activate pytorch-from-scratch
# python resnext/train.py --batch-size 128 --epochs 10 --lr-warmup-epochs 5 --limit-train-iters 100 --limit-val-iters 100
# python resnext/train.py --batch-size 128 --epochs 10 --lr-warmup-epochs 5

accelerate launch resnext/train_accelerate.py \
--output-dir /media/bryan/ssd01/expr/resnext_from_scratch/debug01 \
--train-batch-size 128 --val-batch-size 200 \
--epochs 10 --lr-warmup-epochs 5 --limit-train-iters 500 --limit-val-iters 100