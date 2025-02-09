import torch
import evaluate
import numpy as np

from torchvision.utils import make_grid

from torchvision.transforms.v2.functional import normalize
from genaibook.core import show_images

import matplotlib.pyplot as plt

# Use the 'agg' backend for Matplotlib
plt.switch_backend("agg")


def unnormalize(
    tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    mean = torch.as_tensor(mean, dtype=torch.float)
    std = torch.as_tensor(std, dtype=torch.float)
    unnormalized = normalize(tensor, (-mean / std).tolist(), (1.0 / std).tolist())
    unnormalized = torch.clamp(unnormalized, min=0, max=1)
    return unnormalized


def show_images_unnormalize(
    ims,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    num_images=25,
    max_cols=5,
):
    ims = unnormalize(ims, mean, std)
    num_images = min(ims.shape[0], num_images)
    ncols = min(max_cols, num_images)
    nrows = int(np.ceil(num_images / max_cols))
    return show_images(ims, nrows=nrows, ncols=ncols)


def create_image_grid(
    images,
    pred_labels,
    gt_labels,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_images=25,
    nrow=5,
    label_max_chars=20,
):
    images = images[:max_images]
    pred_labels = pred_labels[:max_images]
    gt_labels = gt_labels[:max_images]

    # Unnormalize images
    images = unnormalize(images, mean, std)

    # Create a grid of images
    grid = make_grid(images, nrow=nrow, padding=0)

    # Plot the grid
    # (width, height)
    fig, ax = plt.subplots(figsize=(15, int(np.ceil(images.shape[0] / nrow)) * 3))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis("off")

    # Add labels
    for i in range(images.shape[0]):
        row = i // nrow
        col = i % nrow
        pred_label = pred_labels[i][:label_max_chars]
        gt_label = gt_labels[i][:label_max_chars]

        plt.text(
            col * images.shape[3],
            row * images.shape[2],
            f"Pr: {pred_label}\nGt: {gt_label}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="square, pad=0"),
            verticalalignment="top",
        )

    # Adjust layout to reduce white space
    fig.tight_layout(pad=0)

    # Save the figure to a numpy array
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # Convert ARGB to RGB
    image_array = image_array[..., [1, 2, 3]]

    plt.close(fig)  # Close the figure to free memory
    return image_array


class CombinedEvaluations(evaluate.CombinedEvaluations):
    """
    Extended this class https://github.com/huggingface/evaluate/blob/v0.4.3/src/evaluate/module.py#L872
    """

    def compute(self, predictions=None, references=None, **kwargs):
        results = []
        kwargs_per_evaluation_module = {
            name: {} for name in self.evaluation_module_names
        }
        for key, value in kwargs.items():
            if key not in kwargs_per_evaluation_module:
                for k in kwargs_per_evaluation_module:
                    kwargs_per_evaluation_module[k].update({key: value})
            elif key in kwargs_per_evaluation_module and isinstance(value, dict):
                kwargs_per_evaluation_module[key].update(value)

        for evaluation_module in self.evaluation_modules:
            batch = {
                "predictions": predictions,
                "references": references,
                **kwargs_per_evaluation_module[evaluation_module.name],
            }
            results.append(evaluation_module.compute(**batch))

        return self._merge_results(results)
