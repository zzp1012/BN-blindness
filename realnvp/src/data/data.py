import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# import internal libs
from utils import get_logger

class DataInfo():
    def __init__(self, 
                 name: str, 
                 channel: int, 
                 size: int):
        """Instantiates a DataInfo.

        Args:
            name: name of dataset.
            channel: number of image channels.
            size: height and width of an image.
        """
        self.name = name
        self.channel = channel
        self.size = size

def load_data(dataset: str,
              root: str = "../data") -> tuple:
    """Load dataset.

    Args:
        dataset: name of dataset.
        root: root directory of dataset.
    Returns:
        a torch dataset and its associated information.
    """
    logger = get_logger(__name__)
    logger.info("Loading dataset %s" % dataset)
    
    if dataset == 'cifar10':    # 3 x 32 x 32
        data_info = DataInfo(dataset, 3, 32)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), 
             transforms.ToTensor()])
        train_set = datasets.CIFAR10(root, 
            train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root,
            train=False, download=True, transform=transforms.ToTensor())
        [train_split, val_split] = [train_set, test_set]
    elif dataset == "mnist":    # 1 x 28 x 28
        data_info = DataInfo(dataset, 1, 28)
        transform = transforms.Compose(
            [transforms.ToTensor()])
        train_set = datasets.MNIST(root,
            train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root, 
            train=False, download=True, transform=transform)
        [train_split, val_split] = [train_set, test_set]
    elif dataset == 'celeba':   # 3 x 218 x 178
        data_info = DataInfo(dataset, 3, 64)
        def CelebACrop(images):
            return transforms.functional.crop(images, 40, 15, 148, 148)
        transform = transforms.Compose(
            [CelebACrop, 
             transforms.Resize(64), 
             transforms.RandomHorizontalFlip(p=0.5), 
             transforms.ToTensor()])
        train_set = datasets.ImageFolder(root, 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [150000, 12770])
    elif dataset == 'imnet32':
        data_info = DataInfo(dataset, 3, 32)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder(root, 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [1250000, 31149])
    elif dataset == 'imnet64':
        data_info = DataInfo(dataset, 3, 64)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder(root, 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [1250000, 31149])
    else:
        raise ValueError('Dataset %s not supported.' % dataset)

    logger.info(f"len(train_split) = {len(train_split)}; len(val_split) = {len(val_split)}")
    logger.info("Dataset %s loaded." % dataset)
    
    return train_split, val_split, data_info