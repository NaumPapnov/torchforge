from torch import cat
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    Sigmoid
)
from torchvision.models import resnet34, ResNet34_Weights


class VGG11(Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
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

class Inception(Module):
    def __init__(self,
                 in_channels: int,
                 cn1x1: int,
                 cn3x3reduce: int,
                 cn3x3: int,
                 cn5x5reduce: int,
                 cn5x5: int,
                 poolcn1x1: int):
        super().__init__()
        self.branch1 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=cn1x1, kernel_size=1, padding = 0),
            BatchNorm2d(cn1x1),
            ReLU()
        )
        self.branch2 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=cn3x3reduce, kernel_size=1, padding = 0),
            BatchNorm2d(cn3x3reduce),
            ReLU(),
            Conv2d(in_channels=cn3x3reduce, out_channels=cn3x3, kernel_size=3, padding = 1),
            BatchNorm2d(cn3x3),
            ReLU()
        )
        self.branch3 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=cn5x5reduce, kernel_size=1, padding = 0),
            BatchNorm2d(cn5x5reduce),
            ReLU(),
            Conv2d(in_channels=cn5x5reduce, out_channels=cn5x5, kernel_size=5, padding = 2),
            BatchNorm2d(cn5x5),
            ReLU()
        )
        self.branch4 = Sequential(
            MaxPool2d(kernel_size=3, stride=1, padding = 1),
            Conv2d(in_channels=in_channels, out_channels=poolcn1x1, kernel_size=1, padding = 0),
            BatchNorm2d(poolcn1x1),
            ReLU()
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        output = cat([branch1, branch2, branch3, branch4], 1)
        return output

class AuxClassifier(Module):
    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.net = Sequential(
            #AvgPool2d(kernel_size=5, stride=3),
            Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1),
            ReLU(),
            Flatten(),
            Linear(2*2*128, 1024),
            Dropout(p=0.7),
            Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        return(x)
    
class GoogleNet(Module):
    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.conv1 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.conv2 = Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(192),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.aux1 = AuxClassifier(512, num_classes)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.aux2 = AuxClassifier(528, num_classes)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        #self.avgpool5 = AvgPool2d(kernel_size=2, stride=1)
        self.dropout = Dropout(p=0.4)
        self.fc = Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        #x = self.avgpool5(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training:
            return x, aux2, aux1
        else:
            return x

class ResNet18StandartBlock(Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            stride: int = 1,
            use_downsample: bool = False
            ):
        super().__init__()
        self.use_downsample = use_downsample
        self.conv1 =  Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                    )
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding
            )
        self.bn2 = BatchNorm2d(out_channels)
        self.downsample = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride
            ),
            BatchNorm2d(out_channels)
            )

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_downsample:
            y = self.downsample(y)
        x += y
        x= self.relu(x)
        return x

class ResNet18(Module):
    def __init__(
            self,
            in_channels: int,
            ):
        super().__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()
        self.layer1 = ResNet18StandartBlock(64, 64, 3, 1, 1)
        self.layer2 = ResNet18StandartBlock(64, 64, 3, 1, 1)
        self.layer3 = ResNet18StandartBlock(64, 128, 3, 1, 2, True)
        self.layer4 = ResNet18StandartBlock(128, 128, 3, 1, 1)
        self.layer5 = ResNet18StandartBlock(128, 256, 3, 1, 2, True)
        self.layer6 = ResNet18StandartBlock(256, 256, 3, 1)
        self.layer7 = ResNet18StandartBlock(256, 512, 3, 1, 2, True)
        self.layer8 = ResNet18StandartBlock(512, 512, 3, 1)
        self.avgpool = AdaptiveAvgPool2d((1,1))
        self.flatten = Flatten()
        self.fc = Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class SimpleObjectDetection(Module):

    def __init__(self, out_features=200, trainable_backbone_layers=2, box_grid=3):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        in_features = backbone.fc.in_features

        # всё до avgpool: на входе 384x384 это карта признаков 512x12x12
        self.trunk = Sequential(*list(backbone.children())[:-2])

        for param in self.trunk.parameters():
            param.requires_grad = False
        res_layers = [self.trunk[4], self.trunk[5], self.trunk[6], self.trunk[7]]
        for layer in res_layers[len(res_layers) - trainable_backbone_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.pool_cls = AdaptiveAvgPool2d(1)
        self.pool_box = AdaptiveAvgPool2d(box_grid)

        self.classification_head = Sequential(
            Flatten(),
            Linear(in_features, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(0.3),
            Linear(512, out_features)
        )

        self.bounding_box_head = Sequential(
            Flatten(),
            Linear(in_features * box_grid * box_grid, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(0.2),
            Linear(512, 256),
            BatchNorm1d(256),
            ReLU(),
            Linear(256, 4),
            Sigmoid()
        )

    def forward(self, x):
        features = self.trunk(x)
        classification_logits = self.classification_head(self.pool_cls(features))
        bbox = self.bounding_box_head(self.pool_box(features))
        return classification_logits, bbox