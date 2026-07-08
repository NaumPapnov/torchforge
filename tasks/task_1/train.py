# here you write script for model training, e.g. loading data, initializing model and trainer, etc.
# save results, checkpoints, etc. in the outputs directory, but dont push model weights to git, instead save them locally and add them to .gitignore
from training.trainer import TrainerArgs, Trainer
from torch.nn import BCELoss
from torch.optim import Adam
from models.classification import SimpleBinaryClassificationNet
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from datasets.classification import ClassificationTableDataset
from torch.utils.data import DataLoader
import torch


def binary_accuracy(predictions, targets):
    predicted_labels = (predictions >= 0.5).float()
    return (predicted_labels == targets).float().mean().item()


def binary_roc_auc(predictions, targets):
    return roc_auc_score(
        targets.detach().cpu().numpy().ravel(),
        predictions.detach().cpu().numpy().ravel(),
    )


def train(model, args, optimizer, train_data_loader, valid_data_loader):
    trainer = Trainer(model, args, optimizer, train_data_loader, valid_data_loader)
    trainer.train()

if __name__ == "__main__":
    

    loss = BCELoss()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

    args = TrainerArgs(
        loss,
        epochs=50,
        device=device,
        checkpoint_step=10,
        metrics={
            "accuracy": binary_accuracy,
            "roc_auc": binary_roc_auc,
        },
    )
    
    X, y = make_classification(n_samples=5000, n_features=20, random_state=42, n_classes=2)
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    train_dataset = ClassificationTableDataset(X_train, y_train)
    valid_dataset = ClassificationTableDataset(X_val, y_val)
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    
    model = SimpleBinaryClassificationNet(input_dim=20)
    optimizer = Adam(model.parameters(), args.learning_rate)
    
    train(model, args, optimizer, train_data_loader, valid_data_loader)
