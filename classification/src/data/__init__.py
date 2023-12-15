import torch
from torch.utils.data import Dataset
from typing import Tuple

def prepare_dataset(dataset: str,
                    root: str = "../data/") -> Tuple[Dataset, Dataset]:
    """prepare the dataset.

    Args:
        dataset (str): the dataset name.
        root (str): the root path of the dataset.

    Returns:
        trainset and testset
    """
    if dataset == "cifar10":
        import data.cifar10 as cifar10
        trainset, testset = cifar10.load(root)
    else:
        raise NotImplementedError(f"dataset {dataset} is not implemented.")
    return trainset, testset