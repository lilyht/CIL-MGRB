import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PretrainedResnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        self.fe = nn.Sequential(*list(self.net.children())[:-1])

    def forward(self, x):
        outputs = self.fe(x)
        outputs = torch.squeeze(outputs)
        return outputs


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        # shortcut

        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Classifier(nn.Module):
    def __init__(self, num_classes=100, temperature=2.0):
        super(Classifier, self).__init__()
        self.T = temperature
        self.num_classes = num_classes
        self.fc1 = nn.Linear(512, num_classes)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x):
        output = self.fc1(x)
        return output


class BiasCorrection(nn.Module):
    """
    Bias Correction Layer
    """

    def __init__(self, num_classes=20):
        super(BiasCorrection, self).__init__()
        self.num_classes = num_classes
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))

    def forward(self, x):
        return self.alpha * x + self.beta

    def printParam(self):
        print(self.alpha.item(), self.beta.item())


# https://github.com/hshustc/CVPR19_Incremental_Learning/blob/HEAD/cifar100-class-incremental/modified_linear.py
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input):
        # w_norm = self.weight.data.norm(dim=1, keepdim=True)
        # w_norm = w_norm.expand_as(self.weight).add_(self.epsilon)
        # x_norm = input.data.norm(dim=1, keepdim=True)
        # x_norm = x_norm.expand_as(input).add_(self.epsilon)
        # w = self.weight.div(w_norm)
        # x = input.div(x_norm)
        out = F.linear(F.normalize(input, p=2, dim=1), \
                       F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinear(nn.Module):
    # consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1)  # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100, complexity=2, temperature=2.0):
        super().__init__()

        self.T = temperature
        self.in_channels = 64
        self.complexity = complexity
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        outputchannel = 128
        if complexity > 0:
            self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
            outputchannel = 256
        if complexity > 1:
            # self.conv5_x = self.conv_layer(256,512)
            self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
            self.conv6_x = self.conv_layer(512, num_classes)  # 最后输出的维度
            outputchannel = 512
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(outputchannel * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def conv_layer(self, input_channel, output_channel, kernel_size=3, padding=1):
        # print("conv layer input", input_channel, "output", output_channel)
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True))
        return res

    def _make_layer(self, block, out_channels, num_blocks, stride):

        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer

        """
        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        # print("Making resnet layer with channel", out_channels, "block", num_blocks, "stride", stride)

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        if self.complexity > 0:
            output = self.conv4_x(output)
        if self.complexity > 1:
            output = self.conv5_x(output)
            # output = self.conv6_x(output)

        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)

        return output


class ResNetWithMultiOutput(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcs = nn.ModuleList([nn.Linear(512 * block.expansion, nc) for nc in num_classes])

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = [self.fcs[i](output) for i in range(len(self.fcs))]
        return tuple(output)


def resnet18(num_class=100):
    """ return a ResNet 18 object
    """
    print("type of num_class: {}".format(type(num_class)))
    if isinstance(num_class, list):
        return ResNetWithMultiOutput(BasicBlock, [2, 2, 2, 2], num_class)
    else:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_class)


def resnet34(num_class=100):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])
