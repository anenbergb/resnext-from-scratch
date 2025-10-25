#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0

conda activate jax
JAX_PLATFORM_NAME=gpu python resnext/train_jax.py \
--output-dir /media/bryan/ssd01/expr/resnext_from_scratch/debug02 \
--batch-size 32 \
--epochs 10 --lr-warmup-epochs 2 --limit-train-iters 100 --limit-val-iters 100