import os
import time
from argparse import ArgumentParser

from monai.config import print_config

import torch

from builders.data_loader import build_data_loaders
from builders.model import build_model

print_config()

data_dir = "./data/FeTS2022"
roi = (128, 128, 128)
feature_size = 48
drop_rate = 0.0
attn_drop_rate = 0.0
dropout_path_rate = 0.0
use_checkpoint = True

partitioning = "natural"
selected_partitions = None
shuffle = True
num_workers = 8

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

    model = build_model(
        img_size=roi,
        feature_size=feature_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
    ).to_device(args.device)

    train_loaders, val_loaders = build_data_loaders(
        data_dir,
        roi,
        batch_size,
        partitioning=partitioning,
        selected_partitions=selected_partitions,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    start = time.monotonic()
    args = parse_args()

    train_model(args)

    end = time.monotonic()

    elapsed = end - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
