import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.utils import save_image, make_grid

# import internal libs
from data.utils import logit_transform
from utils import get_logger, update_dict
from config import SCALE_REG

def create_batches(dataset: Subset,
                   batch_size: int,
                   seed: int,
                   method: str,
                   pos_lbl: int,
                   neg_lbl: int,) -> list:
    """create the batches

    Args:
        dataset: the dataset
        batch_size: the batch size
        seed: the seed
        method: the method to create batches
        pos_lbl: the label of positive samples.
        neg_lbl: the label of negative samples.

    Return:
        the batches
    """
    logger = get_logger(f"{__name__}.create_batches")
    # use dataloader
    inputs, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    logger.debug(f"inputs shape: {inputs.shape}; labels shape: {labels.shape}")
    # create the indices
    if method == "random":
        indices = np.where((labels == neg_lbl) | (labels == pos_lbl))[0]
        random.Random(seed).shuffle(indices)
        batch_indices = np.array_split(indices, len(indices) // batch_size)
    elif method == "label":
        batch_indices = []
        repeat_num = 300
        for itr in range(1, repeat_num+1):
            for i, label in enumerate(range(len(dataset.classes))):
                if label == pos_lbl or label == neg_lbl:    
                    indices = np.where(labels == label)[0]
                    random.Random((seed - 1) * repeat_num + itr + i).shuffle(indices)
                    batch_indices.append(np.array_split(indices, len(indices) // batch_size)[0])
                else:
                    continue
    else:
        raise ValueError(f"unknown method: {method}")
    # create the batches
    batches = []
    for idx in batch_indices:
        batches.append((inputs[idx], labels[idx]))
    return batches


def train(save_path: str,
          device: torch.device,
          data_info,
          pos_lbl: int,
          neg_lbl: int,
          train_split: Subset,
          val_split: Subset,
          flow: nn.Module,
          batch_size: int,
          lr: float,
          momentum: float,
          decay: float,
          weight_decay: float,
          epochs: int,
          sample_size: int,
          method: str,
          scale_reg: float = SCALE_REG) -> None:
    """train realNVP model.

    Args:
        save_path: path to save model.
        device: device to use.
        data_info: data information.
        pos_lbl: the label of positive samples.
        neg_lbl: the label of negative samples.
        train_split: train dataset.
        val_split: validation dataset.
        flow: realNVP model.
        batch_size: batch size.
        lr: learning rate.
        momentum: momentum.
        decay: learning rate decay.
        weight_decay: the weight decay
        epochs: epochs.
        method: the method to create batches.
        sample_size: number of samples to generate.
    
    Returns:
        None
    """
    logger = get_logger(__name__)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # define the dataloader
    val_batches = create_batches(val_split, batch_size, seed=0, 
        method="random", pos_lbl=pos_lbl, neg_lbl=neg_lbl)

    # define the optimizer
    optimizer = optim.Adamax(flow.parameters(), lr=lr, 
        betas=(momentum, decay), weight_decay=weight_decay, eps=1e-7)

    # initial the res_dict
    total_res_dict = {
        "train_loss": [],
        "train_log_ll": [],
        "train_bits_per_dim": [],
        "val_loss": [],
        "val_log_ll": [],
    }

    image_size = data_info.channel * data_info.size**2

    # start
    for epoch in range(1, epochs+1):
        logger.info(f"####Epoch {epoch}...")

        # train
        flow.train()
        train_loss_lst, train_log_ll_lst, train_bit_per_dim_lst = [], [], []
        train_batches = create_batches(train_split, batch_size, 
            epoch, method, pos_lbl, neg_lbl)
        for batch_idx, data_ in enumerate(train_batches, 1):
            x, _ = data_
            x = x.to(device)
            # log-determinant of Jacobian from the logit transform
            x, log_det = logit_transform(x, device)

            # log-likelihood of input minibatch
            log_ll, weight_scale = flow(x)
            log_ll = (log_ll + log_det).mean()

            # add L2 regularization on scaling factors
            loss = -log_ll + scale_reg * weight_scale

            # update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record the loss and log-likelihood
            train_loss_lst.append(loss.item())
            train_log_ll_lst.append(log_ll.item())
            bit_per_dim = (-log_ll.item() + np.log(256.) * image_size) \
                    / (image_size * np.log(2.))
            train_bit_per_dim_lst.append(bit_per_dim)
            if batch_idx % 10 == 0:
                logger.info('[%d/%d]\tloss: %.3f\tlog-ll: %.3f\tbits/dim: %.3f' % \
                    (batch_idx, len(train_batches), loss.item(), log_ll.item(), bit_per_dim))
        
        # take average
        train_loss = sum(train_loss_lst) / len(train_loss_lst)
        train_log_ll = sum(train_log_ll_lst) / len(train_log_ll_lst)
        train_bit_per_dim = sum(train_bit_per_dim_lst) / len(train_bit_per_dim_lst)
        logger.info(f"On avergage - train loss: {train_loss:.3f}\tlog-ll: \
            {train_log_ll:.3f}\tbits/dim: {train_bit_per_dim:.3f}")

        # evaluation
        flow.eval()
        val_loss_lst, val_log_ll_lst, val_bit_per_dim_lst = [], [], []
        with torch.no_grad():
            for batch_idx, data_ in enumerate(val_batches, 1):
                x, _ = data_
                x = x.to(device)
                # log-determinant of Jacobian from the logit transform
                x, log_det = logit_transform(x, device)

                # log-likelihood of input minibatch
                log_ll, weight_scale = flow(x)
                log_ll = (log_ll + log_det).mean()

                # add L2 regularization on scaling factors
                loss = -log_ll + scale_reg * weight_scale

                # record the loss and log-likelihood
                val_loss_lst.append(loss.item())
                val_log_ll_lst.append(log_ll.item())
                bit_per_dim = (-log_ll.item() + np.log(256.) * image_size) \
                    / (image_size * np.log(2.))
                val_bit_per_dim_lst.append(bit_per_dim)

            # sample from the model
            samples = flow.sample(sample_size)
            samples, _ = logit_transform(samples, device=device, reverse=True)
            image_path = os.path.join(save_path, f"samples")
            os.makedirs(image_path, exist_ok=True)
            save_image(make_grid(samples),
                       os.path.join(image_path, f"{epoch}.png"))

        # take average
        val_loss = sum(val_loss_lst) / len(val_loss_lst)
        val_log_ll = sum(val_log_ll_lst) / len(val_log_ll_lst)
        val_bit_per_dim = sum(val_bit_per_dim_lst) / len(val_bit_per_dim_lst)
        logger.info(f"On avergage - val loss: {val_loss:.3f}\tlog-ll: \
            {val_log_ll:.3f}\tbits/dim: {val_bit_per_dim:.3f}")

        res_dict = {
            "train_loss": [train_loss],
            "train_log_ll": [train_log_ll],
            "train_bits_per_dim": [train_bit_per_dim],
            "val_loss": [val_loss],
            "val_log_ll": [val_log_ll],
            "val_bits_per_dim": [val_bit_per_dim],
        }
        total_res_dict = update_dict(res_dict, total_res_dict)

        if epoch % 10 == 0 or epoch == epochs:
            # save the res
            res_df = pd.DataFrame.from_dict(total_res_dict)
            res_df.to_csv(os.path.join(save_path, "train.csv"), index = False)
            # save model
            torch.save(flow.state_dict(), os.path.join(save_path, f"model_epoch{epoch}.pt"))
            logger.info(f'Partial results to {save_path}')            

    logger.info('Training finished at epoch %d.' % epoch)