import os
import time
from argparse import ArgumentParser
from functools import partial

import flwr as fl

from monai.config import print_config
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from monai.utils import MetricReduction

import torch

from builders.data_loader import build_data_loaders
from builders.model import build_model
from client import FlowerClient
from utils.helpers import get_parameters

print_config()

roi = (128, 128, 128)
feature_size = 48
drop_rate = 0.0
attn_drop_rate = 0.0
dropout_path_rate = 0.0
use_checkpoint = True

data_dir = "./data/FeTS2022"
batch_size = 2
partitioning = "natural"
selected_partitions = None
shuffle = True
num_workers = 8

pred_threshold = 0.5

sw_batch_size = 4
fold = 1
infer_overlap = 0.5

lr = 1e-4
weight_decay = 1e-5

num_epochs_per_round = 1
num_rounds = 3
max_epochs = 100

fraction_fit = 0.3
fraction_evaluate = 0.3
min_fit_clients = 3
min_evaluate_clients = 3


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


def run(args):

    cuda = args.cuda and torch.cuda.is_available()
    if cuda:
        args.device = "cuda"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        torch.backends.cudnn.benchmark = True
    else:
        args.device = "cpu"

    model = build_model(
        img_size=roi,
        feature_size=feature_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
    )

    train_loaders, val_loaders, num_partitions = build_data_loaders(
        data_dir,
        roi,
        batch_size,
        partitioning=partitioning,
        selected_partitions=selected_partitions,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    dice_loss = DiceLoss(sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(threshold=pred_threshold)
    dice_acc = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )
    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )

    def client_fn(cid) -> FlowerClient:
        train_loader = train_loaders[int(cid)]
        val_loader = val_loaders[int(cid)]
        return FlowerClient(cid, model, train_loader, optimizer, dice_loss, args.device)

    params = get_parameters(model)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=num_partitions,
        initial_parameters=fl.common.ndarrays_to_parameters(params),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_partitions,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    start = time.monotonic()
    args = parse_args()

    run(args)

    end = time.monotonic()

    elapsed = end - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
