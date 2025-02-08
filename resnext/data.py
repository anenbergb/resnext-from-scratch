from typing import Optional, Callable
import torch
from datasets import load_dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

CORRUPTED_IMAGENET_IMAGES = {
    "train": (
        93403,
        128675,
        132982,
        299807,
        332307,
        390540,
        447450,
        492410,
        537742,
        555333,
        737902,
        868710,
        932734,
        989727,
        1021316,
        1027236,
        1050644,
        1176937,
        1200174,
    ),
    "validation": (),
    "test": (84554, 96055),
}


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, split: str = "train", transform: Optional[Callable] = None):
        assert split in ("train", "validation", "test")

        self.hf_dataset = load_dataset(
            "ILSVRC/imagenet-1k", split=split, trust_remote_code=True
        )
        self.hf_dataset = self.hf_dataset.select(
            [
                i
                for i in range(len(self.hf_dataset))
                if i not in CORRUPTED_IMAGENET_IMAGES[split]
            ]
        )
        self.label_names = self.hf_dataset.features["label"].names
        self.num_classes = len(self.label_names)
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        sample["class_name"] = self.label_names[sample["label"]]
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)
        image = sample["image"].convert("RGB")
        sample["pil_image"] = image
        if self.transform:
            sample["image"] = self.transform(image)
        return sample


"""
Torchvision transforms
- https://pytorch.org/vision/0.21/transforms.html#v1-or-v2-which-one-should-i-use
- https://pytorch.org/vision/0.21/auto_examples/transforms/plot_transforms_getting_started.html


"""


def get_val_transforms(
    resize_size=232,
    crop_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    return v2.Compose(
        [
            v2.ToImage(),  # convert PIL image to tensor
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(resize_size),  # resize the smaller edge of the image. bilinear
            v2.CenterCrop(crop_size),  # square crop
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
            v2.ToPureTensor(),
        ]
    )


def get_train_transforms(
    crop_size=176,
    hflip_prob=0.5,
    random_erase=0.1,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    transforms = [
        v2.ToImage(),  # convert PIL image to tensor
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomResizedCrop(crop_size),  # random crop and resize
    ]
    if hflip_prob > 0:
        transforms.append(v2.RandomHorizontalFlip(hflip_prob))
    transforms += [
        v2.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=mean, std=std),
    ]
    if random_erase > 0:
        transforms.append(v2.RandomErasing(random_erase))

    transforms.append(v2.ToPureTensor())
    return v2.Compose(transforms)


class Collate:
    def __call__(self, batch):
        """
        batch should be a list of dictionaries with keys "pil_image", "label", "image", and "class_name"
        """
        images = [sample["pil_image"] for sample in batch]
        for sample in batch:
            sample.pop("pil_image")

        batch = torch.utils.data.default_collate(batch)
        batch["pil_image"] = images
        return batch


def labels_getter(batch):
    return batch["label"]


class MixUpCutMixCollate:
    """
    How to use CutMix and MixUp https://pytorch.org/vision/0.21/auto_examples/transforms/plot_cutmix_mixup.html
    """

    def __init__(self, num_classes=1000, mixup_alpha=0.2, cutmix_alpha=1.0):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        assert (
            mixup_alpha > 0 or cutmix_alpha > 0
        ), "At least one of mixup_alpha or cutmix_alpha must be greater than 0"
        transforms = []
        if mixup_alpha > 0:
            transforms.append(
                v2.MixUp(
                    alpha=mixup_alpha,
                    num_classes=num_classes,
                    labels_getter=labels_getter,
                )
            )
        if cutmix_alpha > 0:
            transforms.append(
                v2.CutMix(
                    alpha=cutmix_alpha,
                    num_classes=num_classes,
                    labels_getter=labels_getter,
                )
            )
        self.transform = v2.RandomChoice(transforms)

    def __call__(self, batch):
        """
        batch should be a list of dictionaries with keys "pil_image", "image", "label", and "class_name"
        """
        # have to remove the pil_image from the batch before applying the transform
        images = [sample["pil_image"] for sample in batch]
        for sample in batch:
            sample.pop("pil_image")

        batch = torch.utils.data.default_collate(batch)
        batch = self.transform(batch)
        batch["pil_image"] = images
        return batch


def get_collate_function(num_classes=1000, mixup_alpha=0.2, cutmix_alpha=1.0):
    if mixup_alpha > 0 or cutmix_alpha > 0:
        return MixUpCutMixCollate(
            num_classes=num_classes, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha
        )
    else:
        return Collate()


# Repeat augment
# ra_sampler=True,
# ra_reps=4,
