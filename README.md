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
# Example training results

Training the ResNeXt model on the ImageNet-1K dataset across 2x RTX 4090 GPUs for 100 epochs, 5 warm-up epochs, and with 512 batch size per GPU yielded the following results after 1d 7hr. 

| Model Name                             | Acc@1  | Acc@5 |
|----------------------------------------|--------|--------|
| ResNeXt50_32X4D (default model)  | 0.7386  | 0.9203 |

These results do not quite reach the reference accuracies reported by the Pytorch team as cited [on the references page](references.md)
because the model was only trained for 100 epochs, rather than 600, and some of the training recipe optimizations such as
exponential moving average (EMA) and repeat augmentations.

The [ImageNet-1k dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k) is downloaded from Huggingface 
```
huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset
```

# Tensorboard logs
![image](https://github.com/user-attachments/assets/5dff4627-b284-4072-a72d-164d557aec4c)
![image](https://github.com/user-attachments/assets/e50d9b8b-64b3-43e7-811d-e4af1eb2e81b)

# A sampling of predictions from the validation set
(This image grid is automatically rendered to tensorboard during training)
![tensorboard](https://github.com/user-attachments/assets/1857f2fa-cd2c-4385-a1ba-801b179fac4b)
