from torch.nn import Module, CrossEntropyLoss
from torch import no_grad, max, save
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from tqdm.auto import tqdm
from typing import Tuple
class TrainerArgs:
    # arguments for the trainer, e.g. learning rate, batch size, number of epochs, etc.
    def __init__(
            self,
            loss_func,
            optimizer: Optimizer = Adam,
            learning_rate: float = 1e-3,
            batch_size: int = 64,
            device: str = 'cpu',
            epochs: int = 5,
            checkpoint_dir: str = 'output/folder',
            checkpoint_step: int = 1
    ) -> None:
        super().__init__(learning_rate, batch_size, device, loss_func, 
                         optimizer, epochs, checkpoint_dir, checkpoint_step)

class Trainer: 
    # code to train your model, evaluate it, save checkpoints, etc.
    def __init__(
        self,
        model: Module,
        args: TrainerArgs
    ) -> None:
        super().__init__(model, args)

    def _train_one_epoch(
            self,
            model: Module,   
            train_data_loader: DataLoader
    ) -> float:
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in tqdm(train_data_loader, desc='Training'):
            x_batch, y_batch = x_batch.to(self.args.device), y_batch.to(self.args.device)

            predictions = model(x_batch)
            loss = self.args.loss_func(predictions, y_batch)

            total_loss += loss.item()
            optimizer = self.args.optimizer(model.parameters(), self.args.learning_rate)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(train_data_loader)
        return avg_loss

    def evaluate(
            self,
            model, 
            valid_data_loader: DataLoader
    ) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0

        with no_grad():
            for x_batch, y_batch in tqdm(valid_data_loader, desc='Validation'):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                predictions = model(x_batch)
                loss = self.args.loss_func(predictions, y_batch)
                total_loss += loss.item()

                _, predicted_labels = max(predictions, dim=-1)
                total_correct += (predicted_labels == y_batch).sum().item()

            avg_loss = total_loss / len(valid_data_loader)
            accuracy = total_correct / len(valid_data_loader.dataset)
            return avg_loss, accuracy
        
    def train(
            self,
            model: Module, 
            train_data_loader: DataLoader, 
            valid_data_loader: DataLoader
    ):
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(self.args.epochs):
            print(f'\nEpoch {epoch+1}/{self.args.epochs}')

            train_loss = self._train_one_epoch(model, train_data_loader)
            val_loss, val_accur = self.evaluate(model, valid_data_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accur)

            print(f'Epoch {epoch+1} ->'
                f'Train Loss: {train_loss:.4f} |'
                f'Validation Loss: {val_loss:.4f} |'
                f'Validation Accuracy: {val_accur:.4f}')
            
            if (epoch + 1) % self.args.save_every == 0:
                save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accur,
                }, f"checkpoints/model_epoch_{epoch+1}.json")

        return train_losses, val_losses, val_accuracies