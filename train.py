import os
import time
from argparse import ArgumentParser

from monai.config import print_config

import torch

print_config()

roi = (128, 128, 128)
feature_size = 48
drop_rate = 0.0
attn_drop_rate = 0.0
dropout_path_rate = 0.0
use_checkpoint = True

batch_size = 2
sw_batch_size = 4
fold = 1
infer_overlap = 0.5
max_epochs = 100
val_every = 10


def parse_args():
    parser = ArgumentParser(description="Federated Tumor Segmentation")

    """ GPU settings """
    parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU devices. For example: (0,1) or 0",
    )

    args = parser.parse_args()
    return args


def train_model(args):

    cuda = args.cuda and torch.cuda.is_available()
    if cuda:
        args.device = "cuda"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    else:
        args.device = "cpu"


if __name__ == "__main__":
    start = time.monotonic()
    args = parse_args()

    train_model(args)

    end = time.monotonic()

    elapsed = end - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
