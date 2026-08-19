from torch.nn import Module
from torch import Tensor, cat, no_grad, save, tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.auto import tqdm
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass
from utils.metrics_utils import training_plot
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = 'outputs'
MetricFn = Callable[[Tensor, Tensor], float]

@dataclass
class TrainerArgs:
    # arguments for the trainer, e.g. learning rate, batch size, number of epochs, etc.
    loss_func: Module
    loss_bbox: Module | None = None
    loss_bbox_lm: float = 1e-1
    learning_rate: float = 1e-3
    batch_size: int = 64
    device: str = 'cpu'
    epochs: int = 5
    output_dir: str = 'output'
    checkpoint_step: int = 1
    metrics: Optional[Dict[str, MetricFn]] = None
    bbox_metrics: Optional[Dict[str, MetricFn]] = None
    draw_plots: bool = True
         

class Trainer: 
    # code to train your model, evaluate it, save checkpoints, etc.
    def __init__(
        self,
        model: Module,
        args: TrainerArgs,
        optimizer: Optimizer,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        aux: bool = False
    ) -> None:
        self.model = model.to(args.device)
        self.args = args
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.aux = aux

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for x_batch, y_batch in tqdm(
            self.train_data_loader,
            desc='Training',
            leave=False,
            dynamic_ncols=True,
        ):
            x_batch, y_batch = x_batch.to(self.args.device), y_batch.to(self.args.device)

            loss = 0
            if self.aux:
                predictions, *auxes = self.model(x_batch)
                
                loss_main = self.args.loss_func(predictions, y_batch)
                loss_aux = 0.3 * sum(
                    self.args.loss_func(aux_pred, y_batch)
                    for aux_pred in auxes
                )

                loss = loss_main + loss_aux
            else:
                predictions = self.model(x_batch)
                loss = self.args.loss_func(predictions, y_batch)

            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        avg_loss = total_loss / len(self.train_data_loader)
        return avg_loss


    def _train_one_epoch_multi_task(self) -> float:
        self.model.train()
        total_loss = 0
        for x_batch, y_label_batch, y_bbox_batch in tqdm(
            self.train_data_loader,
            desc='Training',
            leave=False,
            dynamic_ncols=True
        ):
            x_batch = x_batch.to(self.args.device)
            y_label_batch = y_label_batch.to(self.args.device)
            y_bbox_batch = y_bbox_batch.to(self.args.device)
            pred_labels, pred_bbox = self.model(x_batch)
            loss_labels = self.args.loss_func(pred_labels, y_label_batch)
            loss_bbox = self.args.loss_bbox(pred_bbox, y_bbox_batch)
            loss = loss_labels + self.args.loss_bbox_lm * loss_bbox

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_data_loader)

        return avg_loss

    def evaluate(
            self
            ) -> Tuple[float, Dict[str, float]] | Tuple[float, Dict[str, float], Dict[str, float]]:
        self.model.eval()
        if self.args.loss_bbox:
            all_pred_labels = []
            all_pred_bbox = []
            all_targets_label = []
            all_targets_bbox = []
        else:
            all_predictions = []
            all_targets = []
        total_loss = 0.0
        

        with no_grad():
            if self.args.loss_bbox:
                for x_batch, y_label_batch, y_bbox_batch in tqdm(
                    self.valid_data_loader,
                    desc='Validation',
                    leave=False,
                    dynamic_ncols=True,
                ):
                    x_batch  = x_batch.to(self.args.device) 
                    y_label_batch = y_label_batch.to(self.args.device)
                    y_bbox_batch = y_bbox_batch.to(self.args.device)

                    pred_labels, pred_bbox = self.model(x_batch)
                    loss_labels = self.args.loss_func(pred_labels, y_label_batch)
                    loss_bbox = self.args.loss_bbox(pred_bbox, y_bbox_batch)
                    total_loss += loss_labels.item() + self.args.loss_bbox_lm * loss_bbox.item()
                    all_pred_labels.append(pred_labels.detach())
                    all_pred_bbox.append(pred_bbox.detach())
                    
                    all_targets_label.append(y_label_batch.detach())
                    all_targets_bbox.append(y_bbox_batch.detach())

                avg_loss = total_loss / len(self.valid_data_loader)

                
                avg_label_metrics = {}
                avg_bbox_metrics = {}
                if self.args.metrics and self.args.bbox_metrics and all_pred_labels and all_pred_bbox:
                    pred_labels = cat(all_pred_labels, dim=0)
                    pred_bbox = cat(all_pred_bbox, dim=0)
                    targets_label = cat(all_targets_label, dim=0)
                    targets_bbox = cat(all_targets_bbox, dim=0)
                    for metric_name, metric_fn in self.args.metrics.items():
                        metric_value = metric_fn(pred_labels, targets_label)
                        if hasattr(metric_value, 'item'):
                            metric_value = metric_value.item()
                        avg_label_metrics[metric_name] = float(metric_value)
                    for metric_name, metric_fn in self.args.bbox_metrics.items():
                            metric_value = metric_fn(pred_bbox, targets_bbox)
                            if hasattr(metric_value, 'item'):
                                metric_value = metric_value.item()
                            avg_bbox_metrics[metric_name] = float(metric_value)                
                return avg_loss, avg_label_metrics, avg_bbox_metrics 
                
            else:
                for x_batch, y_batch in tqdm(
                    self.valid_data_loader,
                    desc='Validation',
                    leave=False,
                    dynamic_ncols=True,
                ):
                    y_batch = tensor(y_batch)
                    x_batch, y_batch = x_batch.to(self.args.device), y_batch.to(self.args.device)

                    predictions = self.model(x_batch)
                    loss = self.args.loss_func(predictions, y_batch)
                    total_loss += loss.item()
                    all_predictions.append(predictions.detach())
                    all_targets.append(y_batch.detach())

                avg_loss = total_loss / len(self.valid_data_loader)
                
                avg_metrics = {}
                if self.args.metrics and all_predictions:
                    predictions = cat(all_predictions, dim=0)
                    targets = cat(all_targets, dim=0)
                    for metric_name, metric_fn in self.args.metrics.items():
                        metric_value = metric_fn(predictions, targets)
                        if hasattr(metric_value, 'item'):
                            metric_value = metric_value.item()
                        avg_metrics[metric_name] = float(metric_value)
                return avg_loss, avg_metrics
        
    def train(self):
        train_losses = []
        val_losses = []
        if self.args.loss_bbox:
            val_label_metrics_history = {name: [] for name in (self.args.metrics or {})}
            val_bbox_metrics_history = {name: [] for name in (self.args.bbox_metrics or {})}
            final_label_metrics_log = ''
            final_bbox_metrics_log = ''
        else:
            val_metrics_history = {name: [] for name in (self.args.metrics or {})}
            final_metrics_log = ''
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)
            
        run_path = f"{self.args.output_dir}/run-{str(datetime.datetime.now()).replace(' ', '_')}"
        
        os.mkdir(run_path)

        epoch_bar = tqdm(range(self.args.epochs), desc='Epochs', dynamic_ncols=True)
        

        for epoch in epoch_bar:
            if self.args.loss_bbox:
                train_loss = self._train_one_epoch_multi_task()
            else:
                train_loss = self._train_one_epoch()
            

            train_losses.append(train_loss)
            if self.args.loss_bbox:
                val_loss, val_label_metrics, val_bbox_metrics = self.evaluate()
                val_losses.append(val_loss)
                for metric_name, metric_value in val_label_metrics.items():
                    val_label_metrics_history[metric_name].append(metric_value)

                metrics_log = ''.join(
                    f'| Validation {metric_name.replace("_", " ").title()}: {metric_value:.4f} '
                    for metric_name, metric_value in val_label_metrics.items()
                )
                final_label_metrics_log = metrics_log

                for metric_name, metric_value in val_bbox_metrics.items():
                    val_bbox_metrics_history[metric_name].append(metric_value)

                metrics_log = ''.join(
                    f'| Validation {metric_name.replace("_", " ").title()}: {metric_value:.4f} '
                    for metric_name, metric_value in val_bbox_metrics.items()
                )
                final_bbox_metrics_log = metrics_log

                epoch_bar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    **{
                        metric_name.lower(): f'{metric_value:.4f}'
                        for metric_name, metric_value 
                        in val_label_metrics.items() | val_bbox_metrics.items()
                    },
                })
            
                if (epoch + 1) % self.args.checkpoint_step == 0:
                    save({
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_label_metrics": val_label_metrics,
                        "val_bbox_mitrics": val_bbox_metrics,
                    }, f"{run_path}/model_epoch_{epoch+1}.pt")
            else:
                val_loss, val_metrics = self.evaluate()
                for metric_name, metric_value in val_metrics.items():
                    val_metrics_history[metric_name].append(metric_value)

                metrics_log = ''.join(
                    f'| Validation {metric_name.replace("_", " ").title()}: {metric_value:.4f} '
                    for metric_name, metric_value in val_metrics.items()
                )
                final_metrics_log = metrics_log

                epoch_bar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    **{
                        metric_name.lower(): f'{metric_value:.4f}'
                        for metric_name, metric_value in val_metrics.items()
                    },
                })
            
                if (epoch + 1) % self.args.checkpoint_step == 0:
                    save({
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                    }, f"{run_path}/model_epoch_{epoch+1}.pt")

        epoch_bar.close()
        print(
            f'Epoch {self.args.epochs} ->'
            f'Train Loss: {train_losses[-1]:.4f} |'
            f'Validation Loss: {val_losses[-1]:.4f} '
        )
        if self.args.loss_bbox:
            print(f'{final_label_metrics_log}')
            print(f'{final_bbox_metrics_log}')
        else:
            print(f'{final_metrics_log}')

        if self.args.draw_plots:
            os.mkdir(f'{run_path}/plots')
            if self.args.bbox_metrics:
                training_plot(train_losses, val_losses, val_label_metrics_history)
                plt.savefig(f'{run_path}/plots/losses_and_label_mterics.png')
                training_plot(train_losses, val_losses, val_bbox_metrics_history)
                plt.savefig(f'{run_path}/plots/losses_and_bbox_mterics.png')
            else:
                training_plot(train_losses, val_losses, val_metrics_history)
                plt.savefig(f'{run_path}/plots/losses_and_mterics.png')
        if self.args.loss_bbox:
            return train_losses, val_losses, val_label_metrics_history, val_bbox_metrics_history
        else:
            return train_losses, val_losses, val_metrics_history
