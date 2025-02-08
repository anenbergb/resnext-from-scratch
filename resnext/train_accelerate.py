import argparse
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm, trange
from dataclasses import dataclass

# Huggingface
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import evaluate

from resnext import ResNeXt
from resnext.data import (
    ImageNetDataset,
    get_val_transforms,
    get_train_transforms,
    get_collate_function,
    Collate,
)
from resnext.torchvision_utils import set_weight_decay
from resnext.utils import CombinedEvaluations


@dataclass
class TrainingConfig:
    output_dir: str
    overwrite_output_dir: bool = True  # overwrite the old model
    resume_from_checkpoint: str = None

    train_batch_size: int = 128
    val_batch_size: int = 256

    epochs: int = 600
    limit_train_iters: int = 0
    limit_val_iters: int = 0

    # Optimizer configuration
    momentum: float = 0.9
    # Linear warmup + CosineAnnealingLR
    lr: float = 0.5  # seems pretty high
    lr_warmup_epochs: int = 5
    lr_warmup_decay: float = 0.01
    lr_min: float = 0.0

    # Regularization and Augmentation
    weight_decay: float = 2e-05
    norm_weight_decay: float = 0.0

    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    ra_sampler: bool = True
    ra_reps: int = 4

    # EMA configuration
    model_ema: bool = True
    model_ema_steps: int = 32
    model_ema_decay: float = 0.99998

    mixed_precision: str = "bf16"  # no for float32

    checkpoint_total_limit: int = 3
    checkpoint_epochs: int = 1
    save_image_epochs: int = 1
    seed: int = 0

    num_workers: int = 2


def train_resnext(config: TrainingConfig):
    """
    Default values from https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
    """

    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        # logging_dir
        automatic_checkpoint_naming=True,
        total_limit=config.checkpoint_total_limit,
        save_on_each_node=False,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(os.path.basename(config.output_dir))

    val_dataset = ImageNetDataset(split="validation", transform=get_val_transforms())
    train_dataset = ImageNetDataset(split="train", transform=get_train_transforms())

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=get_collate_function(
            num_classes=train_dataset.num_classes,
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
        ),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config.num_workers,
        collate_fn=Collate(),
    )

    model = ResNeXt()  # ResNeXt-50 (32x4d)

    parameters = set_weight_decay(
        model,
        config.weight_decay,
        norm_weight_decay=config.norm_weight_decay,
    )

    optimizer = torch.optim.SGD(
        parameters,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.lr_warmup_decay,
        total_iters=config.lr_warmup_epochs,
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs - config.lr_warmup_epochs, eta_min=config.lr_min
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[config.lr_warmup_epochs]
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    if config.resume_from_checkpoint is not None and os.path.exists(
        config.resume_from_checkpoint
    ):
        accelerator.load_state(config.resume_from_checkpoint)
    # TODO: automatically load the most recent checkpoint from the output_dir

    # How do I load from a checkpoint?
    # accelerator.load_state()

    global_step = 0
    for epoch in range(config.epochs):
        total_loss = 0
        model.train()
        for step, batch in (
            progress_bar := tqdm(
                enumerate(train_dataloader),
                total=(
                    len(train_dataloader)
                    if config.limit_train_iters == 0
                    else config.limit_train_iters
                ),
                disable=not accelerator.is_local_main_process,
                desc=f"Epoch {epoch}",
            )
        ):
            if config.limit_train_iters > 0 and step >= config.limit_train_iters:
                break

            optimizer.zero_grad()
            images = batch["image"]
            labels = batch["label"]
            logits = model(images)
            with accelerator.autocast():
                loss = criterion(logits, labels)
            total_loss += loss.detach().item()
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            current_lr = scheduler.get_last_lr()[0]
            logs = {"loss": loss.detach().item(), "lr": current_lr}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if epoch % config.checkpoint_epochs == 0:
            # save_directory = os.path.join(config.output_dir, "checkpoints")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                output_dir = os.path.join(config.output_dir, f"epoch_{epoch}")
                # accelerator.save_model(model, save_directory)
                accelerator.save_state(output_dir)

        scheduler.step()  # once per epoch

        val_metrics = run_validation(
            accelerator,
            model,
            criterion,
            val_dataloader,
            limit_val_iters=config.limit_val_iters,
            global_step=global_step,
        )
        if accelerator.is_main_process:
            val_print_str = f"Validation metrics [Epoch {epoch}]: "
            for k, v in val_metrics.items():
                val_print_str += f"{k}: {v:.3f} "
            accelerator.print(val_print_str)
            log = {f"val/{k}": v for k, v in val_metrics.items()}
            log["epoch"] = epoch
            accelerator.log(log, step=global_step)

    accelerator.end_training()


def run_validation(
    accelerator, model, criterion, val_data_loader, limit_val_iters=0, global_step=0
):

    if accelerator.is_main_process:
        metrics = CombinedEvaluations(["accuracy", "f1", "precision", "recall"])
        total_loss = torch.tensor(0.0, device=accelerator.device)
        total_num_images = torch.tensor(0, dtype=torch.long, device=accelerator.device)

    model.eval()
    with torch.inference_mode():
        for step, batch in tqdm(
            enumerate(val_data_loader),
            total=len(val_data_loader) if limit_val_iters == 0 else limit_val_iters,
            disable=not accelerator.is_local_main_process,
            desc="Validation",
        ):
            if limit_val_iters > 0 and step >= limit_val_iters:
                break
            images = batch["image"]
            labels = batch["label"]
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            with accelerator.autocast():
                loss = criterion(logits, labels)  # average loss

            loss = accelerator.gather(loss * images.size(0))
            num_images = torch.tensor(
                images.size(0), dtype=torch.long, device=accelerator.device
            )
            num_images = accelerator.gather(num_images)
            preds, labels = accelerator.gather_for_metrics((preds, labels))

            if accelerator.is_main_process:
                total_loss += loss.sum()
                total_num_images += num_images.sum()
                metrics.add_batch(predictions=preds, references=labels)

            # log the predictions for the first batch
            # Accelerate tensorboard tracker
            # https://github.com/huggingface/accelerate/blob/main/src/accelerate/tracking.py#L165
            # if accelerator.is_main_process and step == 0:
            #     tensorboard = accelerator.get_tracker("tensorboard")

            #     tensorboard.log_images(step = global_step)

    val_metrics = {}
    if accelerator.is_main_process:
        val_metrics = metrics.compute(
            f1={"average": "macro"},
            precision={"average": "macro", "zero_division": 0},
            recall={"average": "macro", "zero_division": 0},
        )

        avg_loss = (total_loss / total_num_images).item()
        val_metrics["loss"] = avg_loss
    return val_metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Run training loop for ResNeXt model on ImageNet dataset.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/bryan/ssd01/expr/resnext_from_scratch/run01",
        help="Path to save the model",
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=128, help="Training batch size"
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument("--epochs", type=int, default=600, help="Epochs")
    parser.add_argument("--lr-warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument(
        "--limit-train-iters",
        type=int,
        default=0,
        help="Limit number of training iterations per epoch",
    )
    parser.add_argument(
        "--limit-val-iters",
        type=int,
        default=0,
        help="Limit number of val iterations per epoch",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = TrainingConfig(
        output_dir=args.output_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr_warmup_epochs=args.lr_warmup_epochs,
        limit_train_iters=args.limit_train_iters,
        limit_val_iters=args.limit_val_iters,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    sys.exit(train_resnext(config))
