
import torch
import os
from training.trainer import TrainerArgs, Trainer
from models.classification import ResNet18
from torch.nn import CrossEntropyLoss
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from sklearn.metrics import f1_score


def f1(predictions, target):
    return f1_score(
        target.detach().cpu().numpy().ravel(),
        predictions.argmax(dim=1).detach().cpu().numpy().ravel(),
        average='macro'
    )

def train(model, args, optimizer, trainloader, valloader):
    trainer = Trainer(model, args, optimizer, trainloader, valloader)
    trainer.train()

def evaluate(model, args, optimizer, trainloader, testloader):
    trainer = Trainer(model, args, optimizer, trainloader, testloader)
    loss, metrics = trainer.evaluate()
    return loss, metrics

if __name__ == "__main__":
    loss = CrossEntropyLoss()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    args = TrainerArgs(
        loss,
        device=device,
        epochs=20,
        checkpoint_step=5,
        metrics= {
            'f1': f1,
        }
    )
    transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
        ])
    
    if not os.path.exists('./data'):
        os.mkdir('./data')

    dataset = CIFAR10("./data", train=True, transform=transform, download=True)
    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=42,
        stratify=dataset.targets
        )

    trainset = Subset(dataset, train_idx)
    valset = Subset(dataset, val_idx)
    testset = CIFAR10('./data', train=False, transform=transform, download=True)

    trainloader = DataLoader(trainset, args.batch_size, shuffle=True)
    valloader = DataLoader(valset, args.batch_size, shuffle=False)
    testloader = DataLoader(testset, args.batch_size, shuffle=False)

    model = ResNet18(in_channels=3)
    optimizer = Adam(model.parameters(), args.learning_rate)

    train(model, args, optimizer, trainloader, valloader)
    loss, metrics = evaluate(model, args, optimizer, trainloader, testloader)
    print(f'\nTest loss: {loss}, \tTest metrics {metrics}')