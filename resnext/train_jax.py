import argparse
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange

import torch
from resnext.model_jax import ResNeXt
from resnext.data import (
    ImageNetDataset,
    get_val_transforms,
    get_train_transforms,
    get_collate_function,
    Collate,
)
import jax
from flax import nnx
import optax


def train_resnext(
    output_dir,
    batch_size=128,
    epochs=600,
    lr_warmup_epochs=5,
    num_workers=0,
    limit_train_iters=0,
    limit_val_iters=0,
    random_seed=42,
):
    """
    Default values from https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
    """
    momentum = 0.9

    # Linear warmup + CosineAnnealingLR
    lr = 0.5  # seems pretty high
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

    master_key = jax.random.PRNGKey(random_seed)
    init_key, loop_key = jax.random.split(master_key)
    rngs_init = nnx.Rngs(params=init_key)

    val_dataset = ImageNetDataset(split="validation", transform=get_val_transforms())
    train_dataset = ImageNetDataset(split="train", transform=get_train_transforms())

    steps_per_epoch = len(train_dataset) // batch_size

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
            jax=True,
        ),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=Collate(jax=True),
    )

    model = ResNeXt(
        num_classes=train_dataset.num_classes, rngs=rngs_init
    )  # ResNeXt-50 (32x4d)

    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
        init_value=lr * lr_warmup_decay,
        peak_value=lr,
        warmup_steps=lr_warmup_epochs * steps_per_epoch,
        decay_steps=(epochs - lr_warmup_epochs) * steps_per_epoch,
        end_value=lr_min,
    )

    optax_chain = optax.chain(
        # Gradient clipping
        optax.clip(1.0),
        # Applies L2 decay (w = w - weight_decay * w)
        optax.add_decayed_weights(weight_decay=weight_decay),
        optax.sgd(
            learning_rate=lr_schedule,
            momentum=momentum,
        ),
    )
    optimizer = nnx.Optimizer(model, optax_chain, wrt=nnx.Param)

    global_step = 0
    for epoch in (progress := trange(epochs, desc="Training")):
        model.train()
        for step, batch in (
            inner := tqdm(
                enumerate(train_dataloader),
                total=(
                    len(train_dataloader)
                    if limit_train_iters == 0
                    else limit_train_iters
                ),
                desc=f"Epoch {epoch}",
            )
        ):
            if limit_train_iters > 0 and global_step >= limit_train_iters:
                break

            images = batch["image"]  # numpy NHWC
            labels = batch["label"]  # N
            if label_smoothing > 0:
                labels = smooth_one_hot_labels(labels, label_smoothing)

            def loss_fn(model):
                logits = model(images)  # (N,C)
                loss = optax.losses.softmax_cross_entropy(logits, labels)
                avg_loss = jax.numpy.mean(loss)
                return avg_loss

            # It's not necessary to 'zero' the gradients because jax is functional and stateless
            grad_fn = nnx.value_and_grad(loss_fn)
            loss, grads = grad_fn(model)
            optimizer.update(model, grads)

            current_lr = lr_schedule(global_step)
            inner.set_postfix(lr=f"{current_lr:.0e}", loss=f"{float(loss):.3f}")
            global_step += 1

        # accuracy, val_loss = evaluate(
        #     model,
        #     criterion,
        #     val_dataloader,
        #     device=device,
        #     limit_val_iters=limit_val_iters,
        # )

        progress.set_postfix(
            lr=f"{current_lr:.0e}",
            loss=f"{float(loss):.3f}",
            # val_loss=f"{val_loss:.3f}",
            # accuracy=f"{accuracy:.3f}",
        )


# def evaluate(model, criterion, val_data_loader, device="cuda", limit_val_iters=0):
#     all_preds = np.array([], dtype=int)
#     all_labels = np.array([], dtype=int)
#     total_loss = 0
#     model.eval()
#     with torch.inference_mode():
#         for step, batch in (
#             inner := tqdm(
#                 enumerate(val_data_loader),
#                 total=len(val_data_loader) if limit_val_iters == 0 else limit_val_iters,
#                 desc="Validation",
#             )
#         ):
#             if limit_val_iters > 0 and step >= limit_val_iters:
#                 break
#             images = batch["image"].to(device)
#             labels = batch["label"].to(device)
#             logits = model(images)
#             preds = torch.argmax(logits, dim=1).cpu().numpy()
#             all_preds = np.concatenate((all_preds, preds))
#             all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
#             loss = criterion(logits, labels) * images.size(0)
#             total_loss += loss.cpu().item()

#     avg_loss = total_loss / len(all_labels)
#     accuracy = accuracy_score(all_labels, all_preds)
#     return accuracy, avg_loss


def smooth_one_hot_labels(one_hot_labels, smoothing_alpha):
    """Converts integer labels to smoothed one-hot targets (soft labels)."""

    # Calculate the "uniform" component
    uniform_prob = smoothing_alpha / one_hot_labels.shape[-1]

    # Calculate the "confidence" component (1 - alpha)
    confidence = 1.0 - smoothing_alpha

    # Apply the smoothing formula
    smoothed_labels = one_hot_labels * confidence + uniform_prob

    return smoothed_labels


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
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
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
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print("JAX devices: ", jax.devices())
    print("JAX default backend: ", jax.default_backend())

    sys.exit(
        train_resnext(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr_warmup_epochs=args.lr_warmup_epochs,
            limit_train_iters=args.limit_train_iters,
            limit_val_iters=args.limit_val_iters,
            random_seed=args.random_seed,
        )
    )
