from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d


class SimpleRegressionNet(Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear_relu_stack = Sequential(
            Linear(input_dim, 64),
            BatchNorm1d(64),
            ReLU(),
            Linear(64, 128),
            BatchNorm1d(128),
            ReLU(),
            Linear(128, 256),
            BatchNorm1d(256),
            ReLU(),
            Linear(256, 256),
            BatchNorm1d(256),
            ReLU(),
            Linear(256, 256),
            BatchNorm1d(256),
            ReLU(),
            Linear(256, 1)
        )

    def forward(self, X):
        X = self.linear_relu_stack(X)
        return X