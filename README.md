# resnext-from-scratch
Implementation of [ResNeXt neural network](https://arxiv.org/abs/1611.05431) for image classification. 


[Huggingface's Accelerate library](https://huggingface.co/docs/accelerate/en/index) is used in the train_accelerate.py trainer definition.

To launch the multi-gpu distributed trainer
```
accelerate launch resnext/train_accelerate.py \
--output-dir /path/to/output/expr-01 \
--train-batch-size 512 --val-batch-size 512 \
--epochs 100 --lr-warmup-epochs 5
```