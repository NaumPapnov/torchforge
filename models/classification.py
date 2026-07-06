from torch.nn import Module, Sequential, Linear, ReLU, Sigmoid

class SimpleBinaryClassificationNet(Module):
    def __init__(self,):
        super(Module, self).__init__()
        self.linear_relu_stack = Sequential(
            Linear(20, 16),
            ReLU(),
            Linear(16, 16),
            ReLU(),
            Linear(16, 8),
            ReLU(),
            Linear(8, 4),
            ReLU(),
            Linear(4, 2),
            ReLU(),
            Linear(2, 1)
        )
        self.activation = Sigmoid()

    def forward(self, X):
        X = self.linear_relu_stack(X)
        X = self.activation(X)
        return X