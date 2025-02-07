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


@dataclass
class TrainingConfig:
    output_dir: str
    overwrite_output_dir = True  # overwrite the old model

    train_batch_size = 128
    val_batch_size = 256

    epochs = 600
    limit_train_iters = 0
    limit_val_iters = 0

    # Optimizer configuration
    momentum = 0.9
    # Linear warmup + CosineAnnealingLR
    lr = 0.5  # seems pretty high
    lr_warmup_steps = 500
    lr_warmup_decay = 0.01
    lr_min = 0.0

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

    mixed_precision = "bf16"  # no for float32

    save_image_epochs = 1
    save_model_epochs = 1
    seed = 0

    num_workers = 2


def train_resnext(config: TrainingConfig):
    """
    Default values from https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
    """

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("resnext_trainer")

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
    # save the starting state
    accelerator.save_state()  # saves to output_dir/checkpointing/checkpoint_0

    # How do I load from a checkpoint?
    # accelerator.load_state()

    global_step = 0
    for epoch in range(config.epochs):

        num_steps = (
            len(train_dataloader)
            if config.limit_train_iters == 0
            else config.limit_train_iter
        )
        progress_bar = tqdm(
            total=num_steps, disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")
        model.train()
        for step, batch in enumerate(train_dataloader):
            if config.limit_train_iters > 0 and step >= config.limit_train_iters:
                break

            optimizer.zero_grad()
            images = batch["image"]
            labels = batch["label"]
            logits = model(images)
            with accelerator.autocast():
                loss = criterion(logits, labels)
            accelerator.backward()

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            progress_bar.update(1)
            current_lr = scheduler.get_last_lr()[0]
            logs = {"loss": loss.detach().item(), "lr": current_lr, "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if epoch % config.save_model_epochs == 0:
            # save_directory = os.path.join(config.output_dir, "checkpoints")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # accelerator.save_model(model, save_directory)
                # This API is designed to save and resume training states only from within the same python script or training setup
                accelerator.save_state(config.output_dir)

        scheduler.step()  # once per epoch

        accuracy, val_loss = run_validation(
            model,
            criterion,
            val_dataloader,
            device=device,
            limit_val_iters=limit_val_iters,
        )

        progress.set_postfix(
            lr=f"{current_lr:.0e}",
            loss=f"{loss.cpu().item():.3f}",
            val_loss=f"{val_loss:.3f}",
            accuracy=f"{accuracy:.3f}",
        )
    accelerator.end_training()


def run_validation(
    accelerator, model, criterion, val_data_loader, device="cuda", limit_val_iters=0
):
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    total_loss = 0
    model.eval()
    with torch.inference_mode():
        for step, batch in (
            inner := tqdm(
                enumerate(val_data_loader),
                total=len(val_data_loader) if limit_val_iters == 0 else limit_val_iters,
                desc="Validation",
            )
        ):
            if limit_val_iters > 0 and step >= limit_val_iters:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds = np.concatenate((all_preds, preds))
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
            loss = criterion(logits, labels) * images.size(0)
            total_loss += loss.cpu().item()

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, avg_loss


# https://huggingface.co/spaces/evaluate-metric/precision
class ClassificationEvaluator:
    def __init__(self):
        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        results = {
            **self.accuracy.compute(predictions=predictions, references=labels),
            **self.f1.compute(
                predictions=predictions, references=labels, average="macro"
            ),
            **self.precision.compute(
                predictions=predictions,
                references=labels,
                average="macro",
                zero_division=0,
            ),
            **self.recall.compute(
                predictions=predictions,
                references=labels,
                average="macro",
                zero_division=0,
            ),
        }
        return results


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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = TrainingConfig(
        output_dir=args.output_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr_warmup_steps=args.lr_warmup_epochs,
        limit_train_iters=args.limit_train_iters,
        limit_val_iters=args.limit_val_iters,
    )

    sys.exit(train_resnext(config))
