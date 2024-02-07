#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Base trainer class.
"""

import os
import random
import signal
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import plotext as plt
import torch
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm

import wandb
from conf import project as project_conf
from utils import blink_pbar, to_cuda, to_cuda_, update_pbar_str
from utils.helpers import BestNModelSaver
from utils.visualization import visualize_model_predictions


class BaseTrainer:
    def __init__(
        self,
        run_name: str,
        model: torch.nn.Module,
        opt: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_loss: torch.nn.Module,
        accelerator: Accelerator,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        **kwargs,
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
        self._training_loss = training_loss
        self._bps_dim = train_loader.dataset.bps_dim
        self._bps = to_cuda_(train_loader.dataset.bps)
        self._anchor_indices = to_cuda_(train_loader.dataset.anchor_indices)
        self._remap_bps_distances = train_loader.dataset.remap_bps_distances
        self._exponential_map_w = train_loader.dataset.exponential_map_w
        self._n_ctrl_c = 0
        self._viz_n_samples = 1
        self._disable_grad = kwargs.get("disable_grad", False)
        self._ema = None
        self._accelerator = accelerator
        signal.signal(signal.SIGINT, self._terminator)

    @to_cuda
    def _visualize(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        epoch: int,
    ) -> None:
        """Visualize the model predictions.
        Args:
            batch: The batch to process.
            epoch: The current epoch.
        """
        visualize_model_predictions(
            self._model,
            batch,
            epoch,
            bps_dim=self._bps_dim,
            bps=self._bps,
            remap_bps_distances=self._remap_bps_distances,
            exponential_map_w=self._exponential_map_w,
        )  # User implementation goes here (utils/training.py)

    @to_cuda
    def _train_val_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        validation: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training or validation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so write this code only once at the cost of many function calls!
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        samples, labels, _ = batch
        # For this baseline, we onl want one batch dimension so we can reshape all tensors to be (B * T, ...):
        for k, v in samples.items():
            samples[k] = v.view(-1, *v.shape[2:])
        for k, v in labels.items():
            labels[k] = v.view(-1, *v.shape[2:])
        y_hat = self._model(samples["choir"])
        losses = self._training_loss(samples, labels, y_hat)
        loss = sum([v for v in losses.values()])
        return loss, losses

    def _train_epoch(
        self, description: str, visualize: bool, epoch: int, last_val_loss: float
    ) -> float:
        """Perform a single training epoch.
        Args:
            description (str): Description of the epoch for tqdm.
            visualize (bool): Whether to visualize the model predictions.
            epoch (int): Current epoch number.
            last_val_loss (float): Last validation loss.
        Returns:
            float: Average training loss for the epoch.
        """
        epoch_loss, epoch_loss_components = MeanMetric().to(
            self._accelerator.device
        ), defaultdict(MeanMetric)
        self._pbar.reset()
        self._pbar.set_description(description)
        color_code = project_conf.ANSI_COLORS[project_conf.Theme.TRAINING.value]
        has_visualized = 0
        " ==================== Training loop for one epoch ==================== "
        for i, batch in enumerate(self._train_loader):
            if (
                not self._running
                and project_conf.SIGINT_BEHAVIOR
                == project_conf.TerminationBehavior.ABORT_EPOCH
            ):
                self._accelerator.print("[!] Training aborted.")
                break
            self._opt.zero_grad()
            loss, loss_components = self._train_val_iteration(
                batch
            )  # User implementation goes here (train.py)
            if not self._disable_grad:
                self._accelerator.backward(loss)
                self._opt.step()
                if self._ema is not None:
                    self._ema.update()
            epoch_loss.update(loss.clone())
            for k, v in loss_components.items():
                epoch_loss_components[k].to(self._accelerator.device)
                epoch_loss_components[k].update(v.clone())
            update_pbar_str(
                self._pbar,
                f"{description} [loss={epoch_loss.compute():.4f} /"
                + f" val_loss={last_val_loss:.4f}]",
                color_code,
            )
            if (
                visualize
                and has_visualized < self._viz_n_samples
                and (
                    random.Random().random() < 0.15 or i == len(self._train_loader) - 1
                )
            ):
                with torch.no_grad():
                    self._visualize(batch, epoch)
                has_visualized += 1
            self._pbar.update()
        epoch_loss = epoch_loss.compute().clone()
        if project_conf.USE_WANDB:
            wandb.log({"train_loss": epoch_loss}, step=epoch)
            wandb.log(
                {
                    f"Detailed loss - Training/{k}": v.compute().item()
                    for k, v in epoch_loss_components.items()
                },
                step=epoch,
            )
        return epoch_loss

    def _val_epoch(self, description: str, visualize: bool, epoch: int) -> float:
        """Validation loop for one epoch.
        Args:
            description: Description of the epoch for tqdm.
            visualize: Whether to visualize the model predictions.
        Returns:
            float: Average validation loss for the epoch.
        """
        has_visualized = 0
        color_code = project_conf.ANSI_COLORS[project_conf.Theme.VALIDATION.value]
        "==================== Validation loop for one epoch ===================="
        with torch.no_grad():
            val_loss, val_loss_components = MeanMetric().to(
                self._accelerator.device
            ), defaultdict(MeanMetric)
            for i, batch in enumerate(self._val_loader):
                if (
                    not self._running
                    and project_conf.SIGINT_BEHAVIOR
                    == project_conf.TerminationBehavior.ABORT_EPOCH
                ):
                    self._accelerator.print("[!] Training aborted.")
                    break
                # Blink the progress bar to indicate that the validation loop is running
                blink_pbar(i, self._pbar, 4)
                loss, loss_components = self._train_val_iteration(
                    batch, validation=True
                )  # User implementation goes here (train.py)
                val_loss.update(loss.clone())
                for k, v in loss_components.items():
                    val_loss_components[k].to(self._accelerator.device)
                    val_loss_components[k].update(v.clone())
                update_pbar_str(
                    self._pbar,
                    f"{description} [loss={val_loss.compute():.4f} /"
                    + f" min_val_loss={self._model_saver.min_val_loss:.4f}]",
                    color_code,
                )
                " ==================== Visualization ==================== "
                if (
                    visualize
                    and has_visualized < self._viz_n_samples
                    and (
                        random.Random().random() < 0.15
                        or i == len(self._val_loader) - 1
                    )
                ):
                    self._visualize(batch, epoch)
                    has_visualized += 1
            val_loss = val_loss.compute().clone()
            for k, v in val_loss_components.items():
                val_loss_components[k] = v.compute().clone()
            if project_conf.USE_WANDB:
                wandb.log({"val_loss": val_loss}, step=epoch)
                wandb.log(
                    {
                        f"Detailed loss - Validation/{k}": v
                        for k, v in val_loss_components.items()
                    },
                    step=epoch,
                )
            self._model_saver(
                epoch,
                val_loss,
                val_loss_components,
                minimize_metric="distances_from_prior",
            )
            return val_loss

    def train(
        self,
        epochs: int = 10,
        val_every: int = 1,  # Validate every n epochs
        visualize_every: int = 0,  # Visualize every n validation epochs
        visualize_train_every: int = 0,  # Visualize every n training epochs
        visualize_n_samples: int = 1,
        model_ckpt_path: Optional[str] = None,
    ):
        """Train the model for a given number of epochs.
        Args:
            epochs (int): Number of epochs to train for.
            val_every (int): Validate every n epochs.
            visualize_train_every (int): Visualize every n training epochs.
            visualize_every (int): Visualize every n validations.
        Returns:
            None
        """
        if model_ckpt_path is not None:
            self._load_checkpoint(model_ckpt_path)
        if project_conf.PLOT_ENABLED:
            self._setup_plot()
        self._accelerator.print(f"[*] Training for {epochs} epochs")
        self._viz_n_samples = visualize_n_samples
        train_losses, val_losses = [], []
        " ==================== Training loop ==================== "
        for epoch in range(self._epoch, epochs):
            self._epoch = epoch  # Update for the model saver
            if not self._running:
                break
            self._model.train()
            self._pbar.colour = project_conf.Theme.TRAINING.value
            train_losses.append(
                self._train_epoch(
                    f"Epoch {epoch}/{epochs}: Training",
                    visualize_train_every > 0
                    and (epoch + 1) % visualize_train_every == 0,
                    epoch,
                    last_val_loss=val_losses[-1]
                    if len(val_losses) > 0
                    else float("inf"),
                )
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
        self._accelerator.wait_for_everyone()
        self._pbar.close()
        self._save_checkpoint(
            val_losses[-1],
            os.path.join(HydraConfig.get().runtime.output_dir, "last.ckpt"),
        )
        self._accelerator.print(f"[*] Training finished for {self._run_name}!")
        self._accelerator.print(
            f"[*] Best validation loss: {self._model_saver.min_val_loss:.4f} "
            + f"at epoch {self._model_saver.min_val_loss_epoch}."
        )

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
        plt.ylabel("Loss (log scale)")
        plt.yscale("log")
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
        best_metrics = (
            "["
            + ", ".join(
                [
                    f"{metric_name}={metric_value:.2e}"
                    for metric_name, metric_value in self._model_saver.best_metrics.items()
                ]
            )
            + "]"
        )
        plt.scatter(
            [self._model_saver.min_val_loss_epoch],
            [self._model_saver.min_val_loss],
            color="red",
            marker="+",
            label=f"Best model {best_metrics}",
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
        self._accelerator.wait_for_everyone()
        # self._accelerator.save_state(output_dir=ckpt_path.split(".")[0]) # TODO: Always pass a directory to save_checkpoint?
        unwrapped_model = self._accelerator.unwrap_model(self._model)
        unwrapped_opt = self._accelerator.unwrap_model(self._opt)
        unwrapped_sched = self._accelerator.unwrap_model(self._scheduler)
        # torch.save({
        # "val_loss": val_loss,
        # "epoch": self._epoch,
        # **kwargs,
        # }, ckpt_path)
        torch.save(
            {
                **{
                    "model_ckpt": unwrapped_model.state_dict(),
                    "ema_model_ckpt": self._ema.state_dict(),
                    "opt_ckpt": unwrapped_opt.state_dict(),
                    "scheduler_ckpt": unwrapped_sched.state_dict()
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
        self._accelerator.print(
            f"[*] Restoring from checkpoint: {ckpt_path.split('.')[0]}"
        )
        # self._accelerator.load_state(ckpt_path.split(".")[0])
        unwrapped_model = self._accelerator.unwrap_model(self._model)
        unwrapped_opt = self._accelerator.unwrap_model(self._opt)
        unwrapped_sched = self._accelerator.unwrap_model(self._scheduler)
        ckpt = to_cuda_(torch.load(ckpt_path, map_location="cpu"))
        # If the model was optimized with torch.optimize() we need to remove the "_orig_mod"
        # prefix:
        if "_orig_mod" in list(ckpt["model_ckpt"].keys())[0]:
            ckpt["model_ckpt"] = {
                k.replace("_orig_mod.", ""): v for k, v in ckpt["model_ckpt"].items()
            }
        try:
            unwrapped_model.load_state_dict(ckpt["model_ckpt"])
            if self._ema is not None:
                self._ema.load_state_dict(ckpt["ema_model_ckpt"])
            self._accelerator.print("[*] Model weights loaded successfully!")
        except Exception as e:
            if project_conf.PARTIALLY_LOAD_MODEL_IF_NO_FULL_MATCH:
                self._accelerator.print(
                    "[!] Partially loading model weights (no full match between model and checkpoint)"
                )
                self._model.load_state_dict(ckpt["model_ckpt"], strict=False)
            else:
                raise e
        if not model_only:
            unwrapped_opt.load_state_dict(ckpt["opt_ckpt"])
            self._epoch = ckpt["epoch"]
            self._starting_epoch = ckpt["epoch"]
            self._min_val_loss = ckpt["val_loss"]
            if self._scheduler is not None:
                unwrapped_sched.load_state_dict(ckpt["scheduler_ckpt"])

    def _terminator(self, sig, frame):
        """
        Handles the SIGINT signal (Ctrl+C) and stops the training loop.
        """
        if not self._accelerator.is_main_process:
            return
        if (
            project_conf.SIGINT_BEHAVIOR
            == project_conf.TerminationBehavior.WAIT_FOR_EPOCH_END
            and self._n_ctrl_c == 0
        ):
            self._accelerator.print(
                f"[!] SIGINT received. Waiting for epoch to end for {self._run_name}. Press Ctrl+C again to abort."
            )
            self._n_ctrl_c += 1
        elif (
            project_conf.SIGINT_BEHAVIOR == project_conf.TerminationBehavior.ABORT_EPOCH
            or self._n_ctrl_c > 0
        ):
            self._accelerator.print(
                f"[!] SIGINT received. Aborting epoch for {self._run_name}!"
            )
            raise KeyboardInterrupt
        self._running = False
