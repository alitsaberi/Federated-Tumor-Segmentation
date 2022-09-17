from enum import Enum
from typing import List

import numpy as np

import torch


class EnumBase(Enum):
    def __int__(self):
        return self.value

    def __str__(self):
        return self.name.lower()

    @classmethod
    def all(cls):
        return [str(p) for p in cls]

    @classmethod
    def from_str(cls, string):
        return cls[string.upper()]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


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
