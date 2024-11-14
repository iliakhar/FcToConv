from ConvNet import ConvNet
import torch.nn as nn
import torch


class FullConvNet(ConvNet):
    def __init__(self):
        super(FullConvNet, self).__init__()

        # Размер изображений в MNIST - 28x28, кол-во каналов - 1

        self.layer1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=3, stride=1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(10, 15, kernel_size=3, stride=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(15, 3, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(3, 7, kernel_size=12, stride=1), nn.ReLU())
        self.layer5 = nn.Conv2d(7, 10, kernel_size=1, stride=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        return out
