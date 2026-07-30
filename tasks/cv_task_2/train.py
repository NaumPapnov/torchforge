import os

import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from models.classification import GoogleNet
from training.trainer import Trainer, TrainerArgs


def f1(predictions, targets):
    return f1_score(
        targets.detach().cpu().numpy().ravel(),
        predictions.argmax(dim=1).detach().cpu().numpy().ravel(),
        average='macro'
    )

def train(model, args, optimizer, train_data_loader, valid_data_loader):
    trainer = Trainer(model, args, optimizer, train_data_loader, valid_data_loader)
    trainer.train()

def evaluate(model, args, optimizer, train_data_loader, test_data_loader):
    trainer = Trainer(model, args, optimizer, train_data_loader, test_data_loader)
    loss, metrics = trainer.evaluate()
    return loss, metrics

if __name__ == "__main__":
    loss = torch.nn.CrossEntropyLoss()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    args = TrainerArgs(
        loss,
        device=device,
        epochs=20,
        metrics={
            'f1': f1
        }
    )

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
    ])

    if not os.path.exists('./data'):
        os.mkdir('./data') 

    dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
    indices = list(range(len(dataset)))

    train_idx, valid_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=dataset.targets
    )

    trainset = Subset(dataset, train_idx)
    valset = Subset(dataset, valid_idx)
    testset = CIFAR10(root='./data', train=False, transform=transform, download=True)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    model = GoogleNet(input_channels=3)
    optimizer = Adam(model.parameters(), args.learning_rate)

    train(model, args, optimizer, trainloader, valloader)
    loss, metrics = evaluate(model, args, optimizer, trainloader, testloader)
    print(f'\nTest loss: {loss}, \tTest metrics {metrics}')