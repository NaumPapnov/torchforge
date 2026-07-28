from torch.nn import Module, Sequential, Linear, ReLU, Sigmoid
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, MaxPool2d, Softmax
class SimpleBinaryClassificationNet(Module):
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
        self.activation = Sigmoid()

    def forward(self, X):
        X = self.linear_relu_stack(X)
        X = self.activation(X)
        return X
    
    
class VGG11(Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.conv1 = Sequential(
            Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)

        )
        self.conv3 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = Sequential(
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = Sequential(
            Linear(2*2*512, 4096),
            BatchNorm1d(4096),
            Linear(4096, 4096),
            BatchNorm1d(4096),
            Linear(4096, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        #print(f'\nSize of x: {x.shape}')
        x = self.fc(x)
        return x
