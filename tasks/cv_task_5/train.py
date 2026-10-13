import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet34_Weights
from torchvision.ops import box_iou
from torchvision.transforms import v2

from datasets.classification import ObjectDetectionDataset
from models.classification.cv import SimpleObjectDetection
from training.trainer import Trainer, TrainerArgs
from utils.data_utils import show_images

# официальный архив CUB_200_2011.tgz разворачивается именно в такую структуру
DATA_ROOT = './data/CUB_200_2011/CUB_200_2011'


def f1(pred, target):
    return f1_score(
        target.detach().cpu().numpy().ravel(),
        pred.argmax(dim=1).detach().cpu().numpy().ravel(),
        average='macro'
    )

def IoU(pred, target):
    iou = box_iou(pred, target)
    return iou.diag().mean()


def train(model, args, optimizer, train_data_loader, valid_data_loader):
    trainer = Trainer(model, args, optimizer, train_data_loader, valid_data_loader)
    trainer.train()

def evaluate(model, args, optimizer, train_data_loader, test_data_loader):
    trainer = Trainer(model, args, optimizer, train_data_loader, test_data_loader)
    loss, label_metrics, bbox_metrics = trainer.evaluate()
    return loss, label_metrics, bbox_metrics


def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    loss_labels = CrossEntropyLoss()
    # beta=1.0 на нормированных координатах превращает SmoothL1 в чистый L2,
    # у которого градиент затухает вместе с ошибкой — берём порог по масштабу задачи
    loss_bbox = SmoothL1Loss(beta=0.05)
    args = TrainerArgs(
        loss_labels,
        loss_bbox,
        # при lm=0.2 вклад bbox в общий лосс был ~0.2% против CrossEntropy на 200 классов
        loss_bbox_lm=5.0,
        device=device,
        epochs=20,
        checkpoint_step=5,
        batch_size=32,
        metrics={
            'f1': f1,
        },
        bbox_metrics={
            'IoU': IoU,
        }
    )

    transform = v2.Compose([
        v2.Resize((384, 384)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=ResNet34_Weights.DEFAULT.transforms().mean,
            std=ResNet34_Weights.DEFAULT.transforms().std
        )
    ])
    # на валидации и тесте аугментаций быть не должно
    eval_transform = v2.Compose([
        v2.Resize((384, 384)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=ResNet34_Weights.DEFAULT.transforms().mean,
            std=ResNet34_Weights.DEFAULT.transforms().std
        )
    ])

    def build_dataset(tf):
        return ObjectDetectionDataset(
            f'{DATA_ROOT}/image_class_labels.txt',
            f'{DATA_ROOT}/images',
            f'{DATA_ROOT}/images.txt',
            f'{DATA_ROOT}/bounding_boxes.txt',
            transform=tf
        )

    dataset = build_dataset(transform)
    eval_dataset = build_dataset(eval_transform)

    split = pd.read_csv(
        f'{DATA_ROOT}/train_test_split.txt',
        sep=' ',
        names=["image_id", "is_train"]
    )

    train_ids = split.loc[split["is_train"] == 1, "image_id"]

    train_ids, valid_ids = train_test_split(
        train_ids,
        test_size=0.1,
        random_state=42
    )
    id_to_idx = {
        image_id: idx
        for idx, image_id in enumerate(dataset.img_labels.index)
    }

    test_ids = split.loc[split["is_train"] == 0, "image_id"]

    train_ids = [id_to_idx[i] for i in train_ids]
    valid_ids = [id_to_idx[i] for i in valid_ids]
    test_ids = [id_to_idx[i] for i in test_ids]

    trainset = Subset(dataset, train_ids)
    valset = Subset(eval_dataset, valid_ids)
    testset = Subset(eval_dataset, test_ids)

    loader_kwargs = dict(num_workers=8, pin_memory=True, persistent_workers=True)
    trainloader = DataLoader(trainset, args.batch_size, shuffle=True, **loader_kwargs)
    valloader = DataLoader(valset, args.batch_size, shuffle=False, **loader_kwargs)
    testloader = DataLoader(testset, args.batch_size, shuffle=False, **loader_kwargs)

    model = SimpleObjectDetection()

    # дообучаемому backbone нужен заметно меньший шаг, чем головам с нуля
    backbone_params = [p for p in model.trunk.parameters() if p.requires_grad]
    head_params = [
        *model.classification_head.parameters(),
        *model.bounding_box_head.parameters(),
    ]
    optimizer = Adam([
        {'params': backbone_params, 'lr': args.learning_rate * 0.1},
        {'params': head_params, 'lr': args.learning_rate},
    ])

    train(model, args, optimizer, trainloader, valloader)
    final_loss, label_metrics, bbox_metrics = evaluate(model, args, optimizer, trainloader, testloader)
    print(f'Loss: {final_loss}, Label_metrics: {label_metrics}, Bbox_metrics: {bbox_metrics}')

    model.eval()
    all_pred_labels = []
    all_pred_bboxes = []
    with torch.no_grad():
        for images, labels, bboxes in testloader:
            images = images.to(device)
            pred_labels, pred_bboxes = model(images)

            all_pred_labels.append(pred_labels.cpu())
            all_pred_bboxes.append(pred_bboxes.cpu())
    pred_labels = torch.cat(all_pred_labels)
    pred_bboxes = torch.cat(all_pred_bboxes)

    display_transform = v2.Compose([
        v2.Resize((384, 384)),
        v2.ToDtype(torch.float32, scale=True)
    ])

    display_dataset = build_dataset(display_transform)

    display_testset = Subset(display_dataset, test_ids)
    indices = np.random.choice(
        len(display_testset),
        15,
        replace=False
    )
    label_titles = pd.read_csv(
        f'{DATA_ROOT}/classes.txt',
        sep=' ',
        names=["image_id", "class"])
    label_titles = label_titles['class']
    show_images(display_testset, pred_labels, pred_bboxes, label_titles, indices)
    plt.savefig('tasks/cv_task_5/images_with_bboxes')


if __name__ == "__main__":
    main()
