# ResNeXt References

Benchmarks
* https://pytorch.org/vision/main/models.html#classification

Accuracies on ImageNet-1K using single crops:

| Model Name                             | Acc@1  | Acc@5  | Params | GFLOPS |
|----------------------------------------|--------|--------|--------|--------|
| ResNeXt101_32X8D_Weights.IMAGENET1K_V2 | 82.834 | 96.228 | 88.8M  | 16.41  |
| ResNeXt50_32X4D_Weights.IMAGENET1K_V2  | 81.198 | 95.34  | 25.0M  | 4.23   |
| ResNet152_Weights.IMAGENET1K_V2        | 82.284 | 96.002 | 60.2M  | 11.51  |
| ResNet101_Weights.IMAGENET1K_V2        | 81.886 | 95.78  | 44.5M  | 7.8    |
| ResNet50_Weights.IMAGENET1K_V2         | 80.858 | 95.434 | 25.6M  | 4.09   |

The Pytorch team achieved significant increases in accuracy by improving the training recipe [as described here](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/)

Optimizations that worked:
* LR Optimizations - Cosine Schedule
* Data Augmentation
    * Trivial Augment https://arxiv.org/pdf/2103.10158
    * Random Erasing at 10% - regularization effect
    * Mixup and Cutmix - regularization effect by softening the labels and images.
    * Repeated Augmentation - https://arxiv.org/abs/1901.09335, (MultiGrain) https://arxiv.org/abs/1902.05509
* Long training - increasing from 90 epochs to 600 epochs. Increasing the training duration improves performance because of the LR optimizations and strong Augmentation strategies.
* Label Smoothing of 0.1 - stops the model from becoming over confident by softening the ground truth.
* Weight Decay Tuning - regularization applied to all model parameters. They disable weigth decay for the normalization layers.
* Increase image resolution for validation - FixRes. https://arxiv.org/abs/1906.06423. In practice, reduce the training resolution from 224 to 176 such that the relative resolution for validation images is increased.
* Exponential Moving Average (EMA) - increases accuracy of model without increasing its complexity.
* Inference Resize Tuning - At inference time, resize the image from 224x224 to 232x232 and then take central crop

Optimizations that didn't work:
* Optimizers - vanilla SGD with momentum worked as well as other more complex optimizers such as Adam, RMSProp, etc.
* LR Schedulers - Cosine Schedule worked fine and doesn't require tuning additional hyper-parameters
* Automatic Augmetnations - didn't work any better than "Trivial Augment"
* Interpolation - Using bicubic or nearest interpolation didn’t provide significantly better results than bilinear.
* Normalization layers - Using Sync Batch Norm didn’t yield significantly better results than using the regular Batch Norm.


## Baseline ResNet50 Training Recipe
```python
# Optimizer & LR scheme
ngpus=8,
batch_size=32,  # per GPU

epochs=90, 
opt='sgd',  
momentum=0.9,

lr=0.1, 
lr_scheduler='steplr', 
lr_step_size=30, 
lr_gamma=0.1,

# Regularization
weight_decay=1e-4,

# Resizing
interpolation='bilinear', 
val_resize_size=256, 
val_crop_size=224, 
train_crop_size=224,
```

## Updated ResNet50 Training Recipe

```python
# Optimizer & LR scheme
  ngpus=8,
  batch_size=128,  # per GPU

  epochs=600, 
  opt='sgd',  
  momentum=0.9,

  lr=0.5, 
  lr_scheduler='cosineannealinglr', 
  lr_warmup_epochs=5, 
  lr_warmup_method='linear', 
  lr_warmup_decay=0.01, 


  # Regularization and Augmentation
  weight_decay=2e-05, 
  norm_weight_decay=0.0,

  label_smoothing=0.1, 
  mixup_alpha=0.2, 
  cutmix_alpha=1.0, 
  auto_augment='ta_wide', 
  random_erase=0.1, 
  
  ra_sampler=True,
  ra_reps=4,


  # EMA configuration
  model_ema=True, 
  model_ema_steps=32, 
  model_ema_decay=0.99998, 


  # Resizing
  interpolation='bilinear', 
  val_resize_size=232, 
  val_crop_size=224, 
  train_crop_size=176,
```


```
torchrun --nproc_per_node=8 train.py --model resnet50 --batch-size 128 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 --ra-sampler --ra-reps 4
```


## ResNeXt target implementation for this repo is ResNeXt50_32X4D_Weights
* https://pytorch.org/vision/main/models/generated/torchvision.models.resnext50_32x4d.html#torchvision.models.ResNeXt50_32X4D_Weights


| Model Name                             | Acc@1  | Acc@5  | Params | GFLOPS |
|----------------------------------------|--------|--------|--------|--------|
| ResNeXt50_32X4D_Weights.IMAGENET1K_V1  | 77.618 | 93.698 | 25.0M  | 4.23   |
| ResNeXt50_32X4D_Weights.IMAGENET1K_V2  | 81.198 | 95.34  | 25.0M  | 4.23   |

Minor details
* category 997 is omitted
Inference transforms
* The images are resized to resize_size=[232] using interpolation=InterpolationMode.BILINEAR
* Followed by a central crop of crop_size=[224]
* Tensor values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

## Incorporated ideas from the following resources
* Torchvision classification reference https://github.com/pytorch/vision/blob/main/references/classification/train.py
* Huggingface Accelerator https://huggingface.co/docs/accelerate/index
  * examples https://github.com/huggingface/accelerate/blob/main/examples/complete_cv_example.py
  * Accelerator class https://github.com/huggingface/accelerate/blob/main/src/accelerate/accelerator.py
* Evaluate https://huggingface.co/docs/evaluate/index
  * EvaluationModule / Metric class https://github.com/huggingface/evaluate/blob/v0.4.3/src/evaluate/module.py
  * Metric definitions https://github.com/huggingface/evaluate/tree/v0.4.3/metrics

