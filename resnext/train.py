import argparse
import sys
import torch
from resnext import ResNeXt
from resnext.data import (
    ImageNetDataset,
    get_val_transforms,
    get_train_transforms,
    get_collate_function,
    Collate,
)


def train_resnext(
    output_dir,
    batch_size=128,
    epochs=600,
    lr_warmup_epochs=5,
    num_workers=0,
):
    """
    Default values from https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
    """
    opt = "sgd"
    momentum = 0.9

    lr = 0.5  # seems pretty high
    lr_scheduler = "cosineannealinglr"

    lr_warmup_method = "linear"
    lr_warmup_decay = 0.01

    # Regularization and Augmentation
    weight_decay = 2e-05
    norm_weight_decay = 0.0

    label_smoothing = 0.1
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    ra_sampler = True
    ra_reps = 4

    # EMA configuration
    model_ema = True
    model_ema_steps = 32
    model_ema_decay = 0.99998

    val_dataset = ImageNetDataset(split="validation", transform=get_val_transforms())
    train_dataset = ImageNetDataset(split="train", transform=get_train_transforms())

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=get_collate_function(
            num_classes=train_dataset.num_classes,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
        ),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=Collate(),
    )

    model = ResNeXt()  # ResNeXt-50 (32x4d)

    # for batch in train_data_loader:
    #     logits = model(imgs, targets)
    #     # Put your training logic here

    #     print(f"{[img.shape for img in imgs] = }")
    #     print(f"{[type(target) for target in targets] = }")
    #     for name, loss_val in loss_dict.items():
    #         print(f"{name:<20}{loss_val:.3f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Run training loop for ResNeXt model on ImageNet dataset.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/bryan/ssd01/expr/resnext_from_scratch/run01",
        help="Path to save the model",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=600, help="Epochs")
    parser.add_argument("-lr-warmup-epochs", type=int, default=5, help="Warmup epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(
        train_resnext(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr_warmup_epochs=args.lr_warmup_epochs,
        )
    )
