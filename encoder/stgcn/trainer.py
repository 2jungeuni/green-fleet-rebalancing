import torch
import torch.nn as nn
import torch.optim as optim

import time

from config import cfg

class STGCNTrainer:
    """
    This class encapsulates the training and infrence steps
    for the STGCN model.
    """
    def __init__(self,
                 model: nn.Module,
                 loss_fn,
                 optimizer,
                 scheduler,
                 device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.count = 0                      # reset for every epoch
        self.train_loss_val = 0             # reset for every epoch
        self.copy_loss_val = 0              # reset for every epoch

        # save at every epoch
        self.train_loss_dict = {
            "train_loss": [],
            "copy_loss": [],
        }

        self.model.train()
        self.device = device

    def train_step(self,
                   x: torch.Tensor):
        self.optimizer.zero_grad()

        y_hat = self.model(x[:, :cfg.n_hist, :, :])

        copy_l = self.loss_fn(
            x[:, cfg.n_hist - 1: cfg.n_hist, :, :],
            x[:, cfg.n_hist: cfg.n_hist + 1, :, :]
        )

        train_l = self.loss_fn(
            y_hat,
            x[:, cfg.n_hist: cfg.n_hist + 1, :, :]
        )

        train_l.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        return y_hat, train_l, copy_l

    def train(self,
              epoch):
        self.scheduler.step()

        self.train_loss_dict["train_loss"].append(self.train_loss_val / self.count)
        self.train_loss_dict["copy_loss"].append(self.copy_loss_val / self.count)

        self.train_loss_val = 0
        self.copy_loss_val = 0
        self.count = 0

    def save(self,
             epoch,
             step,
             save_file):
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_file)