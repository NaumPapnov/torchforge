import os
from typing import Any, Union

import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from torch import FloatTensor, tensor, long
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
from torchvision.tv_tensors import BoundingBoxes


# implementstion of synthetic dataset for classification task, which inherits from the base dataset class
class ClassificationTableDataset(Dataset):
    def __init__(
            self,
            data: Union[Series, DataFrame, ndarray, tensor],
            targets: Union[Series, DataFrame, ndarray, tensor],
    ) -> None:
        self.data = FloatTensor(data)
        self.targets = FloatTensor(targets).view(-1, 1)

    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        features, target = self.data[index], self.targets[index]

        return features, target

class ObjectDetectionDataset(Dataset):
    def __init__(
            self,
            img_labels: str,
            img_dir: str,
            img_titles: str,
            img_bbox: str,
            transform=None
    ):
        self.img_labels = pd.read_csv(img_labels, sep=' ', index_col=0, header=None)
        self.img_dir = img_dir
        self.img_titles = pd.read_csv(img_titles, sep=' ', index_col=0, header=None)
        self.img_bbox = pd.read_csv(img_bbox, sep=' ', index_col=0, header=None)
        self.img_labels.index = self.img_labels.index.astype(int)
        self.img_titles.index = self.img_titles.index.astype(int)
        self.img_bbox.index = self.img_bbox.index.astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_titles.iloc[idx, 0])
        image = decode_image(img_path, mode=ImageReadMode.RGB)
        _, height, width = image.shape
        label = tensor(self.img_labels.iloc[idx].item() - 1, dtype=long)
        x, y, w, h = self.img_bbox.iloc[idx].values
        x_min, x_max, y_min, y_max = x , x + w, y, y + h
        bbox = BoundingBoxes(
            [x_min, y_min, x_max, y_max],
            format="XYXY",
            canvas_size=(height, width)
        )
        bbox = bbox.reshape(4)
        bbox[0::2] /= width
        bbox[1::2] /= height
        if self.transform:
            image, bbox = self.transform(image, bbox)
        return image, label, bbox
