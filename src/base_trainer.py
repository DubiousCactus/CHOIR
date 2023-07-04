#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Base trainer class.
"""

import signal
from typing import List, Optional, Tuple, Union

import plotext as plt
import torch
import wandb
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm

from conf import project as project_conf
from utils import blink_pbar, to_cuda, update_pbar_str
from utils.helpers import BestNModelSaver
from utils.training import visualize_model_predictions


class BaseTrainer:
    def __init__(
        self,
        run_name: str,
        model: torch.nn.Module,
        opt: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        """Base trainer class.
        Args:
            model (torch.nn.Module): Model to train.
            opt (torch.optim.Optimizer): Optimizer to use.
            train_loader (torch.utils.data.DataLoader): Training dataloader.
            val_loader (torch.utils.data.DataLoader): Validation dataloader.
        """
        self._run_name = run_name
        self._model = model
        self._opt = opt
        self._scheduler = scheduler
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._epoch = 0
        self._starting_epoch = 0
        self._running = True
        self._model_saver = BestNModelSaver(
            project_conf.BEST_N_MODELS_TO_KEEP, self._save_checkpoint
        )
        self._pbar = tqdm(total=len(self._train_loader), desc="Training")
        signal.signal(signal.SIGINT, self._terminator)

    @to_cuda
    def _train_val_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
    ) -> torch.Tensor:
        """Training or validation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so write this code only once at the cost of many function calls!
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        # x, y = batch
        # y_hat = self._model(x)
        # return torch.nn.functional.mse_loss(y_hat, y)
        raise NotImplementedError

    def _train_epoch(self, description: str, epoch: int) -> float:
        """Perform a single training epoch.
        Args:
            description (str): Description of the epoch for tqdm.
            epoch (int): Current epoch number.
        Returns:
            float: Average training loss for the epoch.
        """
        epoch_loss = MeanMetric()
        self._pbar.reset()
        self._pbar.set_description(description)
        color_code = project_conf.ANSI_COLORS[project_conf.Theme.TRAINING.value]
        " ==================== Training loop for one epoch ==================== "
        for batch in self._train_loader:
            if (
                not self._running
                and project_conf.SIGINT_BEHAVIOR
                == project_conf.TerminationBehavior.ABORT_EPOCH
            ):
                print("[!] Training aborted.")
                break
            self._opt.zero_grad()
            loss = self._train_val_iteration(
                batch
            )  # User implementation goes here (train.py)
            loss.backward()
            self._opt.step()
            epoch_loss.update(loss.item())
            update_pbar_str(
                self._pbar,
                f"{description} [loss={epoch_loss.compute():.4f} /"
                + f" min_val_loss={self._model_saver.min_val_loss:.4f}]",
                color_code,
            )
            self._pbar.update()
        epoch_loss = epoch_loss.compute().item()
        if project_conf.USE_WANDB:
            wandb.log({"train_loss": epoch_loss}, step=epoch)
        return epoch_loss

    def _val_epoch(self, description: str, visualize: bool, epoch: int) -> float:
        """Validation loop for one epoch.
        Args:
            description: Description of the epoch for tqdm.
            visualize: Whether to visualize the model predictions.
        Returns:
            float: Average validation loss for the epoch.
        """
        "==================== Validation loop for one epoch ===================="
        color_code = project_conf.ANSI_COLORS[project_conf.Theme.VALIDATION.value]
        with torch.no_grad():
            val_loss = MeanMetric()
            for i, batch in enumerate(self._val_loader):
                if (
                    not self._running
                    and project_conf.SIGINT_BEHAVIOR
                    == project_conf.TerminationBehavior.ABORT_EPOCH
                ):
                    print("[!] Training aborted.")
                    break
                # Blink the progress bar to indicate that the validation loop is running
                blink_pbar(i, self._pbar, 4)
                loss = self._train_val_iteration(
                    batch
                )  # User implementation goes here (train.py)
                val_loss.update(loss.item())
                update_pbar_str(
                    self._pbar,
                    f"{description} [loss={val_loss.compute():.4f} /"
                    + f" min_val_loss={self._model_saver.min_val_loss:.4f}]",
                    color_code,
                )
                " ==================== Visualization ==================== "
                if visualize:
                    visualize_model_predictions(
                        self._model, batch, epoch
                    )  # User implementation goes here (utils/training.py)
            val_loss = val_loss.compute().item()
            if project_conf.USE_WANDB:
                wandb.log({"val_loss": val_loss}, step=epoch)
            self._model_saver(epoch, val_loss)
            return val_loss

    def train(
        self,
        epochs: int = 10,
        val_every: int = 1,  # Validate every n epochs
        visualize_every: int = 10,  # Visualize every n validations
        model_ckpt_path: Optional[str] = None,
    ):
        """Train the model for a given number of epochs.
        Args:
            epochs (int): Number of epochs to train for.
            val_every (int): Validate every n epochs.
            visualize_every (int): Visualize every n validations.
        Returns:
            None
        """
        if model_ckpt_path is not None:
            self._load_checkpoint(model_ckpt_path)
        if project_conf.PLOT_ENABLED:
            self._setup_plot()
        print(f"[*] Training for {epochs} epochs")
        train_losses, val_losses = [], []
        " ==================== Training loop ==================== "
        for epoch in range(self._epoch, epochs):
            self._epoch = epoch  # Update for the model saver
            if not self._running:
                break
            self._model.train()
            self._pbar.colour = project_conf.Theme.TRAINING.value
            train_losses.append(
                self._train_epoch(f"Epoch {epoch}/{epochs}: Training", epoch)
            )
            if epoch % val_every == 0:
                self._model.eval()
                self._pbar.colour = project_conf.Theme.VALIDATION.value
                val_losses.append(
                    self._val_epoch(
                        f"Epoch {epoch}/{epochs}: Validation",
                        visualize_every > 0 and (epoch + 1) % visualize_every == 0,
                        epoch,
                    )
                )
            if self._scheduler is not None:
                self._scheduler.step()
            " ==================== Plotting ==================== "
            if project_conf.PLOT_ENABLED:
                self._plot(epoch, train_losses, val_losses)
        self._pbar.close()
        print(f"[*] Training finished for {self._run_name}!")

    def _setup_plot(self):
        """Setup the plot for training and validation losses."""
        plt.title("Training and validation losses")
        plt.theme("dark")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, True)

    def _plot(self, epoch: int, train_losses: List[float], val_losses: List[float]):
        """Plot the training and validation losses.
        Args:
            epoch (int): Current epoch number.
            train_losses (List[float]): List of training losses.
            val_losses (List[float]): List of validation losses.
        Returns:
            None
        """
        plt.clf()
        plt.theme("dark")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, True)
        plt.plot(
            list(range(self._starting_epoch, epoch + 1)),
            train_losses,
            color=project_conf.Theme.TRAINING.value,
            label="Training loss",
        )
        plt.plot(
            list(range(self._starting_epoch, epoch + 1)),
            val_losses,
            color=project_conf.Theme.VALIDATION.value,
            label="Validation loss",
        )
        plt.scatter(
            [self._model_saver.min_val_loss_epoch],
            [self._model_saver.min_val_loss],
            color="red",
            marker="+",
            label="Best model",
            style="inverted",
        )
        plt.show()

    def _save_checkpoint(self, val_loss: float, ckpt_path: str, **kwargs) -> None:
        """Saves the model and optimizer state to a checkpoint file.
        Args:
            val_loss (float): The validation loss of the model.
            ckpt_path (str): The path to the checkpoint file.
            **kwargs: Additional dictionary to save. Use the format {"key": state_dict}.
        Returns:
            None
        """
        torch.save(
            {
                **{
                    "model_ckpt": self._model.state_dict(),
                    "opt_ckpt": self._opt.state_dict(),
                    "scheduler_ckpt": self._scheduler.state_dict()
                    if self._scheduler is not None
                    else None,
                    "epoch": self._epoch,
                    "val_loss": val_loss,
                },
                **kwargs,
            },
            ckpt_path,
        )

    def _load_checkpoint(self, ckpt_path: str, model_only: bool = False) -> None:
        """Loads the model and optimizer state from a checkpoint file. This method should remain in
        this class because it should be extendable in classes inheriting from this class, instead
        of being overwritten/modified. That would be a source of bugs and a bad practice.
        Args:
            ckpt_path (str): The path to the checkpoint file.
            model_only (bool): If True, only the model is loaded (useful for BaseTester).
        Returns:
            None
        """
        print(f"[*] Restoring from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        try:
            self._model.load_state_dict(ckpt["model_ckpt"])
        except Exception:
            if project_conf.PARTIALLY_LOAD_MODEL_IF_NO_FULL_MATCH:
                print(
                    "[!] Partially loading model weights (no full match between model and checkpoint)"
                )
                self._model.load_state_dict(ckpt["model_ckpt"], strict=False)
        if not model_only:
            self._opt.load_state_dict(ckpt["opt_ckpt"])
            self._epoch = ckpt["epoch"]
            self._starting_epoch = ckpt["epoch"]
            self._min_val_loss = ckpt["val_loss"]
            if self._scheduler is not None:
                self._scheduler.load_state_dict(ckpt["scheduler_ckpt"])

    def _terminator(self, sig, frame):
        """
        Handles the SIGINT signal (Ctrl+C) and stops the training loop.
        """
        if (
            project_conf.SIGINT_BEHAVIOR
            == project_conf.TerminationBehavior.WAIT_FOR_EPOCH_END
        ):
            print("[!] SIGINT received. Waiting for epoch to end.")
        elif (
            project_conf.SIGINT_BEHAVIOR == project_conf.TerminationBehavior.ABORT_EPOCH
        ):
            print("[!] SIGINT received. Aborting epoch.")
        self._running = False
