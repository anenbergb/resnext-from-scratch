from setuptools import setup, find_packages

setup(
    name="resnext-from-scratch",
    version="1.0.0",
    url="https://github.com/anenbergb/resnext-from-scratch",
    author="Bryan Anenberg",
    author_email="anenbergb@gmail.com",
    description="From scratch implentation of ResNeXt image classifier",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorboard",
        "matplotlib",
        "tqdm",
        "types-tqdm",
        "pillow",
        "types-Pillow",
        "datasets",
        "accelerate",
        "scikit-learn",
    ],
    extras_require={
        "torch": [
            "torch",
            "torchvision",
        ],
        "notebook": [
            "jupyter",
            "itkwidgets",
            "jupyter_contrib_nbextensions",
        ],
        # conda install ipykernel
        "dev": ["black", "mypy", "flake8", "isort", "ipdb"],
    },
)