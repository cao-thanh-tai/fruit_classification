from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms


def build_datasets(
    train_dir: Path,
    test_dir: Path,
    image_size: int,
    validation_ratio: float,
    seed: int,
) -> Tuple[Dataset, Dataset, Dataset, list[str]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    full_train = datasets.ImageFolder(train_dir, transform=train_transform)
    evaluation_train = datasets.ImageFolder(
        train_dir,
        transform=evaluation_transform,
    )
    test_dataset = datasets.ImageFolder(test_dir, transform=evaluation_transform)
    if full_train.classes != evaluation_train.classes != test_dataset.classes:
        raise ValueError("Train and test class names/order do not match.")

    validation_size = round(len(full_train) * validation_ratio)
    training_size = len(full_train) - validation_size
    indices = range(len(full_train))
    train_indices, validation_indices = random_split(
        indices,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_dataset = Subset(full_train, train_indices.indices)
    validation_dataset = Subset(
        evaluation_train,
        validation_indices.indices,
    )
    return train_dataset, validation_dataset, test_dataset, full_train.classes
