from torch.nn import Module
from torch import no_grad, max, save
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.auto import tqdm
from typing import Tuple
from dataclasses import dataclass



@dataclass
class TrainerArgs:
    # arguments for the trainer, e.g. learning rate, batch size, number of epochs, etc.
    loss_func: Module
    learning_rate: float = 1e-3
    batch_size: int = 64
    device: str = 'cpu'
    epochs: int = 5
    checkpoint_dir: str = 'output/folder'
    checkpoint_step: int = 1
         

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
        for x_batch, y_batch in tqdm(self.train_data_loader, desc='Training'):
            x_batch, y_batch = x_batch.to(self.args.device), y_batch.to(self.args.device)

            predictions = self.model(x_batch)
            loss = self.args.loss_func(predictions, y_batch)

            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        avg_loss = total_loss / len(self.train_data_loader)
        return avg_loss

    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0

        with no_grad():
            for x_batch, y_batch in tqdm(self.valid_data_loader, desc='Validation'):
                x_batch, y_batch = x_batch.to(self.args.device), y_batch.to(self.args.device)

                predictions = self.model(x_batch)
                loss = self.args.loss_func(predictions, y_batch)
                total_loss += loss.item()

                _, predicted_labels = max(predictions, dim=-1)
                total_correct += (predicted_labels == y_batch).sum().item()

            avg_loss = total_loss / len(self.valid_data_loader)
            accuracy = total_correct / len(self.valid_data_loader.dataset)
            return avg_loss, accuracy
        
    def train(self):
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(self.args.epochs):
            print(f'\nEpoch {epoch+1}/{self.args.epochs}')

            train_loss = self._train_one_epoch()
            val_loss, val_accur = self.evaluate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accur)

            print(f'Epoch {epoch+1} ->'
                f'Train Loss: {train_loss:.4f} |'
                f'Validation Loss: {val_loss:.4f} |'
                f'Validation Accuracy: {val_accur:.4f}')
            
            if (epoch + 1) % self.args.checkpoint_step == 0:
                save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accur,
                }, f"{self.args.checkpoint_dir}/model_epoch_{epoch+1}.pt")

        return train_losses, val_losses, val_accuracies