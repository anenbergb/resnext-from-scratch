"""
Script adopted from https://huggingface.co/datasets/aharley/rvl_cdip/discussions/2
"""

import logging
import warnings
from datasets import load_dataset
from tqdm import trange


def validate_download():
    dataset = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)
    splits = ["train", "validation", "test"]

    logging.basicConfig(
        filename="./validate_download.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Treat all warnings as errors
    warnings.filterwarnings("error")

    for split in splits:
        logger.info(f"Validating all images of split '{split}'...")
        ds = dataset[split]

        for idx in trange(len(ds)):
            try:
                ds[idx]["image"].load()
                ds[idx]["image"].close()
            except Exception as e:
                logger.error(f"{idx}: {e}")

    # No longer treat warnings as errors
    warnings.resetwarnings()


if __name__ == "__main__":
    validate_download()
