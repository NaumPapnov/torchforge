from torch.nn import Module
from torch import Tensor, cat, no_grad, save
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.auto import tqdm
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass
from utils.metrics_utils import training_plot
import os
import datetime
import matplotlib.pyplot as plt


OUTPUT_PATH = 'outputs'
MetricFn = Callable[[Tensor, Tensor], float]

@dataclass
class TrainerArgs:
    # arguments for the trainer, e.g. learning rate, batch size, number of epochs, etc.
    loss_func: Module
    learning_rate: float = 1e-3
    batch_size: int = 64
    device: str = 'cpu'
    epochs: int = 5
    output_dir: str = 'output'
    checkpoint_step: int = 1
    metrics: Optional[Dict[str, MetricFn]] = None
    draw_plots: bool = True
         

class Trainer: 
    # code to train your model, evaluate it, save checkpoints, etc.
    def __init__(
        self,
        model: Module,
        args: TrainerArgs,
        optimizer: Optimizer,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader
    ) -> None:
        self.model = model.to(args.device)
        self.args = args
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

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

            predictions = self.model(x_batch)
            loss = self.args.loss_func(predictions, y_batch)

            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        avg_loss = total_loss / len(self.train_data_loader)
        return avg_loss

    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with no_grad():
            for x_batch, y_batch in tqdm(
                self.valid_data_loader,
                desc='Validation',
                leave=False,
                dynamic_ncols=True,
            ):
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
        val_metrics_history = {name: [] for name in (self.args.metrics or {})}
        
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)
            
        run_path = f"{self.args.output_dir}/run-{str(datetime.datetime.now()).replace(' ', '_')}"
        
        os.mkdir(run_path)

        epoch_bar = tqdm(range(self.args.epochs), desc='Epochs', dynamic_ncols=True)
        final_metrics_log = ''

        for epoch in epoch_bar:
            train_loss = self._train_one_epoch()
            val_loss, val_metrics = self.evaluate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
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
            f'{final_metrics_log}'
        )

        if self.args.draw_plots:
            os.mkdir(f'{run_path}/plots')
            training_plot(train_losses, val_losses, val_metrics_history)
            plt.savefig(f'{run_path}/plots/losses_and_mterics.png')

        return train_losses, val_losses, val_metrics_history
