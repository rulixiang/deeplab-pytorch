import torch
import torch.nn as nn
import torch.nn.functional as F

expansion = 4

class ConvBN(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, batch_norm=None):
        super(ConvBN, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)

        if batch_norm is None:
            self.batch_norm = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=1e-3)
            #self.batch_norm = nn.SyncBatchNorm(out_planes, eps=1e-5, momentum=1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_planes=None, out_planes=None, stride=None, dilation=None, downsample=False):
        super(Bottleneck, self).__init__()
        mid_planes = out_planes // expansion
        
        self.conv1 = ConvBN(in_planes, mid_planes, kernel_size=1, stride=stride, padding=0, dilation=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = ConvBN(mid_planes, mid_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = ConvBN(mid_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1)
        self.relu3 = nn.ReLU(inplace=True)
        if downsample:
            self.shortcut = ConvBN(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, dilation=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        #identity = x
        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out += self.shortcut(x)

        out = self.relu3(out)
        return out

def _make_layer(blocks, in_planes, out_planes, stride, dilation):

    layers = []

    layers.append(Bottleneck(in_planes=in_planes, out_planes=out_planes, stride=stride, dilation=dilation, downsample=True))

    for _ in range(1, blocks):
        layers.append(Bottleneck(in_planes=out_planes, out_planes=out_planes, stride=1, dilation=dilation, downsample=False))

    return nn.Sequential(*layers)

class Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """
    def __init__(self, out_planes):
        super(Stem, self).__init__()
        self.add_module("conv1", ConvBN(in_planes=3, out_planes=out_planes, kernel_size=7, stride=2, padding=3, dilation=1))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))

class ResNet101(nn.Sequential):

    def __init__(self, n_classes, n_blocks):
        super(ResNet101, self).__init__()
        planes = [64 * 2 ** i for i in range(6)]
        self.add_module('layer1', Stem(planes[0]))
        self.add_module('layer2', _make_layer(blocks=n_blocks[0], in_planes=planes[0],out_planes=planes[2], stride=1, dilation=1))
        self.add_module('layer3', _make_layer(blocks=n_blocks[1], in_planes=planes[2],out_planes=planes[3], stride=2, dilation=1))
        self.add_module('layer4', _make_layer(blocks=n_blocks[2], in_planes=planes[3],out_planes=planes[4], stride=2, dilation=1))
        self.add_module('layer5', _make_layer(blocks=n_blocks[3], in_planes=planes[4],out_planes=planes[5], stride=2, dilation=1))

if __name__ == "__main__":
    model = ResNet101(n_classes=1000, n_blocks=[3, 4, 23, 3])
    model.eval()
    image = torch.randn(1, 3, 224, 224)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)