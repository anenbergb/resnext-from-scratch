#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0

conda activate jax
# JAX_DISABLE_JIT=1
JAX_PLATFORM_NAME=gpu python resnext/train_jax.py \
--output-dir /media/bryan/ssd01/expr/resnext_from_scratch/jax \
--batch-size 256 \
--epochs 100 --lr-warmup-epochs 5
# --limit-train-iters 1000 --limit-val-iters 100