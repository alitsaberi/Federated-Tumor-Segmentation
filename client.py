from typing import List

import flwr as fl
import numpy as np

import torch

from utils.helpers import AverageMeter


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, optimizer, loss_func, device):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        epoch_losses = train(
            self.model,
            self.train_loader,
            self.optimizer,
            self.loss_func,
            self.device,
            epochs=config["local_epochs"],
        )
        return (
            get_parameters(self.model),
            len(self.train_loader),
            {"loss": epoch_losses[-1]},
        )

    def evaluate(self, parameters, config):
        pass


def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.Tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def train(
    model,
    train_loader,
    optimizer,
    loss_func,
    device,
    epochs: int,
):
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        run_loss = AverageMeter()
        for idx, batch_data in enumerate(train_loader):
            data, target = batch_data["image"].to(device), batch_data[
                "label"
            ].to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = loss_func(logits, target)
            loss.backward()
            optimizer.step()
            run_loss.update(loss.item(), n=len(batch_data))

        epoch_losses.append(run_loss.avg)

    return epoch_losses
