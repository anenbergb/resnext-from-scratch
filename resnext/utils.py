import torch
import evaluate
import numpy as np

from torchvision.transforms.v2.functional import normalize
from genaibook.core import show_images


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
