import torch
import os
import kagglehub

from torchvision.models import resnet18
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss
from training.trainer import TrainerArgs, Trainer
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def f1(predictions, y_true):
    return f1_score(
        y_true.detach().cpu().numpy().ravel(),
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
    loss = CrossEntropyLoss()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    args = TrainerArgs(
        loss_func=loss,
        device=device,
        epochs=20,
        checkpoint_step=5,
        metrics={
            'f1': f1,
        }
    )

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(256)
    ])

    transform_test = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256)),
        v2.ToDtype(torch.float32, scale=True)
    ])



    path = kagglehub.dataset_download("puneet6060/intel-image-classification")

    train_path = os.path.join(path, "./seg_train/seg_train")
    test_path = os.path.join(path, "./seg_test/seg_test")

    dataset = ImageFolder(train_path, transform=transform)

    indices = list(range(len(dataset)))

    train_idx, valid_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42
    )

    trainset = Subset(dataset, train_idx)
    valset = Subset(dataset, valid_idx)
    testset = ImageFolder(test_path, transform=transform_test)

    trainloader = DataLoader(trainset, args.batch_size, shuffle=True)
    valloader = DataLoader(valset, args.batch_size, shuffle=False)
    testloader = DataLoader(testset, args.batch_size, shuffle=False)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    targets = os.listdir(train_path)
    num_targets = len(targets)
    model.fc = torch.nn.Linear(num_ftrs, num_targets)

    optimizer = Adam(model.fc.parameters(), args.learning_rate)

    train(model, args, optimizer, trainloader, valloader)

    loss, metrics = evaluate(model, args, optimizer, trainloader, testloader)
    print(f'After tail train --- Test loss: {loss}, Test metrics: {metrics}')

    for param in model.parameters():
            param.requires_grad = True

    optimizer = Adam(model.parameters(), args.learning_rate)

    train(model, args, optimizer, trainloader, valloader)
    
    loss, metrics = evaluate(model, args, optimizer, trainloader, testloader)
    print(f'After all model train --- Test loss: {loss}, Test metrics: {metrics}')