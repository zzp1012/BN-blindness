import torch.nn as nn

# import internal libs
from utils import get_logger

def prepare_model(model_name: str,
                  dataset: str,
                  bn_type: str) -> nn.Module:
    """prepare the random initialized model according to the name.

    Args:
        model_name (str): the model name
        dataset (str): the dataset name
        bn_type (str): the Batch normalization type

    Return:
        the model
    """
    logger = get_logger(__name__)
    logger.info(f"prepare the {model_name} model for dataset {dataset}")
    if dataset == "cifar10":
        num_classes = 10
        if model_name.startswith("vgg"):
            import model.cifar_vgg as cifar_vgg
            model = cifar_vgg.__dict__[model_name](num_classes=num_classes, bn_type=bn_type)
        elif model_name.startswith("ResNet"):
            import model.cifar_resnet as cifar_resnet
            model = cifar_resnet.__dict__[model_name](num_classes=num_classes, bn_type=bn_type)
        elif model_name == "densenet_cifar" or model_name.startswith("DenseNet"):
            import model.cifar_densenet as cifar_densenet
            model = cifar_densenet.__dict__[model_name](bn_type=bn_type)
        else:
            raise ValueError(f"unknown model name: {model_name} for dataset {dataset}")
    else:
        raise ValueError(f"{dataset} is not supported.")
    return model
