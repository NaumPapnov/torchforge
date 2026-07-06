from torch.utils.data import Dataset
from typing import Optional, Callable, Any
from pandas import DataFrame, Series

# implementstion of synthetic dataset for classification task, which inherits from the base dataset class
class ClassificationTableDataset(Dataset):
    def __init__(
            self,
            data: DataFrame,
            targets: Series,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(data, targets, transform, target_transform)

    def __len__(self) -> int:
        return len(self.target)
    
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        features, target = self.data[index], self.targets[index]
        if self.transform is not None:
            features = self.transform(features)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return features, target