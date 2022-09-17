import flwr as fl

from utils.helpers import get_parameters, set_parameters, train


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, optimizer, loss_func, device):
        self.cid = cid
        self.model = model.to_device(device)
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
