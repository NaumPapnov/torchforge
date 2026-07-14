from torch.utils.data import Dataset
from typing import Any, Union
from pandas import DataFrame, Series
from numpy import ndarray
from torch import tensor, FloatTensor

# implementstion of synthetic dataset for classification task, which inherits from the base dataset class
class RegressionTableDataset(Dataset):
    def __init__(
            self,
            data: Union[Series, DataFrame, ndarray, tensor],
            targets: Union[Series, DataFrame, ndarray, tensor]
    ) -> None:
        self.data = FloatTensor(data.to_numpy(copy=True))
        self.targets = FloatTensor(targets.to_numpy(copy=True)).view(-1, 1)

    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        features, target = self.data[index], self.targets[index]

        return features, target