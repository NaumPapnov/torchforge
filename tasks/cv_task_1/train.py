import torch
import os
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset
from training.trainer import TrainerArgs, Trainer
from sklearn.metrics import f1_score
from models.classification import VGG11
from torch.optim import Adam
from sklearn.model_selection import train_test_split



def f1(predictions, targets):
    return f1_score(
        targets.detach().cpu().numpy().ravel(),
        predictions.argmax(dim=1).detach().cpu().numpy().ravel(),
        average='macro'
    )

def train(model, args, optimizer, train_data_loader, valid_data_loader):
    trainer = Trainer(model, args, optimizer, train_data_loader, valid_data_loader)
    trainer.train()
    
def evaluate(model, args, optimizer, train_data_loader, valid_data_loader):
    trainer = Trainer(model, args, optimizer, train_data_loader, valid_data_loader)
    loss, metric = trainer.evaluate()
    print(f'Test loss: {loss}, Test metrics: {metric}')

if __name__ == "__main__":
    loss = torch.nn.CrossEntropyLoss()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    # print(device)
    args = TrainerArgs(
        loss,
        epochs=20,
        device=device,
        checkpoint_step=5,
        metrics={
            "f1": f1
        },
    )

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
    ])

    if not os.path.exists('./data'):
        os.mkdir('./data')

    dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
    testset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=dataset.targets
    )

    trainset = Subset(dataset, train_idx)
    valset = Subset(dataset, val_idx)
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    model = VGG11(input_channels=3)
    optimizer = Adam(model.parameters(), args.learning_rate)

    train(model, args, optimizer, trainloader, valloader)
    evaluate(model, args, optimizer, trainloader, testloader)