import os
import argparse

# import internal libs
from data import load_data
from model import build_model
from train import train
from utils import set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from config import DATE, MOMENT, SRC_PATH

def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="Real NVP PyTorch implementation")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the random seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    ## the dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to be modeled.')
    parser.add_argument('--pos', type=int, default=1,
                        help='positive label.')
    parser.add_argument('--neg', type=int, default=0,
                        help='negative label.')                    
    ## model settings
    parser.add_argument('--base_dim', type=int, default=64,
                        help='features in residual blocks of first few layers.')
    parser.add_argument('--res_blocks', type=int, default=0,
                        help='number of residual blocks per group.')
    parser.add_argument('--bottleneck', type=int, default=0,
                        help='whether to use bottleneck in residual blocks.')
    parser.add_argument('--skip', type=int, default=0,
                        help='whether to use skip connection in coupling layers.')
    parser.add_argument('--weight_norm', type=int, default=0,
                        help='whether to apply weight normalization.')
    parser.add_argument('--coupling_bn', type=int, default=0,
                        help='whether to apply batchnorm after coupling layers.')
    parser.add_argument('--affine', type=int, default=1,
                        help='whether to use affine coupling.')
    parser.add_argument('--bn_type', type=str, default='bn',
                        help='type of bn.')
    ## training settings
    parser.add_argument('--method', type=str, default='random',
                        help='method to create batches.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum number of training epoches.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of images in a mini-batch.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='beta1 in Adam optimizer.')
    parser.add_argument('--decay', type=float, default=0.999,
                        help='beta2 in Adam optimizer.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay in Adam optimizer.')
    ## sampling settings
    parser.add_argument('--sample_size', type=int, default=64,
                        help='number of images to generate.')
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([DATE, 
                         MOMENT,
                         f"{args.dataset}",
                         f"{args.pos}And{args.neg}",
                         f"seed{args.seed}",
                         f"blocks{args.res_blocks}",
                         f"bottle{args.bottleneck}",
                         f"{args.bn_type}",
                         f"{args.method}",
                         f"lr{args.lr}",
                         f"bs{args.batch_size}",
                         f"wd{args.wd}"])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args

def main():
    # get the args.
    args = add_args()
    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, args.verbose)
    # set the seed
    set_seed(args.seed)
    # set the device
    args.device = set_device(args.device)

    # show the args.
    logger.info("#########parameters settings....")
    import config
    log_settings(args, config.__dict__)

    # save the current src
    save_current_src(save_path = args.save_path, 
                     src_path = SRC_PATH)

    # prepare the dataset
    logger.info("#########preparing dataset....")
    train_split, val_split, data_info = load_data(dataset = args.dataset)

    # prepare the model
    logger.info("#########preparing model....")
    flow = build_model(device = args.device,
                       data_info = data_info,
                       base_dim = args.base_dim,
                       res_blocks = args.res_blocks,
                       bottleneck = args.bottleneck,
                       skip = args.skip,
                       weight_norm = args.weight_norm,
                       coupling_bn = args.coupling_bn,
                       affine = args.affine,
                       bn_type = args.bn_type)
    logger.info(flow)

    # train
    logger.info("#########training....")
    train(save_path = os.path.join(args.save_path, "train"),
          device = args.device,
          data_info = data_info,
          pos_lbl = args.pos,
          neg_lbl = args.neg,
          train_split = train_split,
          val_split = val_split,
          flow = flow,
          batch_size = args.batch_size,
          lr = args.lr,
          momentum = args.momentum,
          decay = args.decay,
          weight_decay = args.wd,
          epochs = args.epochs,
          method = args.method,
          sample_size = args.sample_size)


if __name__ == "__main__":
    main()