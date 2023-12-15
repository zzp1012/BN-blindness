import os
import argparse
import torch

# import internal libs
from data import prepare_dataset
from model import prepare_model
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
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the random seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--resume_path", default=None, type=str,
                        help='the path of pretrained model.')
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help='the dataset name.')
    parser.add_argument("--method", default="random", type=str,
                        help='the method to train')
    parser.add_argument("--model", default="vgg11", type=str,
                        help='the model name.')
    parser.add_argument("--bn_type", default="none", type=str,
                        help='the Batch normalizaition type.')
    parser.add_argument('--epochs', default=100, type=int,
                        help="set epoch number")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="set the learning rate.")
    parser.add_argument("--bs", default=128, type=int,
                        help="set the batch size")
    parser.add_argument("--wd", default=0.01, type=float,
                        help="set the weight decay")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="set the momentum rate")    
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([DATE, 
                         MOMENT,
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.model}",
                         f"bn_{args.bn_type}",
                         f"{args.method}",
                         f"epochs{args.epochs}",
                         f"lr{args.lr}",
                         f"bs{args.bs}",
                         f"wd{args.wd}",
                         f"momentum{args.momentum}"])
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
    trainset, testset = prepare_dataset(args.dataset)

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset, args.bn_type)
    if args.resume_path:
        logger.info(f"load the pretrained model from {args.resume_path}")
        model.load_state_dict(torch.load(args.resume_path))
    logger.info(model)

    # train the model
    logger.info("#########training model....")
    train(save_path = os.path.join(args.save_path, "exp"),
          device = args.device,
          model = model,
          trainset = trainset,
          testset = testset,
          epochs = args.epochs,
          lr = args.lr,
          batch_size = args.bs,
          weight_decay = args.wd,
          momentum = args.momentum,
          seed = args.seed,
          method = args.method)

if __name__ == "__main__":
    main()