import torch.nn as nn
import torch.nn.functional as F

'''
Reference: Wang et al. "Coarse-to-Fine: Progressive Knowledge Transfer Based Multi-Task 
           Convolutional Neural Network for Intelligent Large-Scale Fault Diagnosis"
           https://github.com/armstrongwang20/PKT-MCNN
'''

class Net0(nn.Module):
    def __init__(self, num_classes):
        super(Net0, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(67968, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net0HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net0HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_c1 = nn.Linear(67968, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(67968, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2


class Net1(nn.Module):
    def __init__(self, num_classes):
        super(Net1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
            nn.Conv2d(128, 256, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(118784, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net1HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net1HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # nn.Conv2d(64, 128, 3, 1),
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=2, stride=2),

        self.fc_c1 = nn.Linear(118784, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(118784, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2


class Net2(nn.Module):
    def __init__(self, num_classes):
        super(Net2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(153216, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net2HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net2HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_c1 = nn.Linear(153216, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(153216, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2

class BH_Net3(nn.Module):
    def __init__(self, num_classes):
        super(BH_Net3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # nn.Conv2d(64, 128, 3, 1),
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        self.fc1 = nn.Linear(107520, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x
    
class Net3(nn.Module):
    def __init__(self, num_classes):
        super(Net3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # nn.Conv2d(64, 128, 3, 1),
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        self.fc1 = nn.Linear(172032, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net3HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net3HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_f1 = nn.Linear(172032, 512)
        self.fc_f2 = nn.Linear(512, num_C_cls)
        # self.dropout_f = nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        # x2 = F.softmax(self.fc_f2(x2), dim=1)
        x2 = self.fc_f1(x)
        # x2 = self.dropout_f(x2)

        return x2


class Net4(nn.Module):
    def __init__(self, num_classes):
        super(Net4, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(16128, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net4HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net4HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_c1 = nn.Linear(16128, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(16128, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2


class Net5(nn.Module):
    def __init__(self, num_classes):
        super(Net5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(21504, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net5HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net5HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_c1 = nn.Linear(21504, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(21504, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2

class BH_Net6(nn.Module):
    def __init__(self, num_classes):
        super(BH_Net6, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc1 = nn.Linear(46080, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

class Net6(nn.Module):
    def __init__(self, num_classes):
        super(Net6, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc1 = nn.Linear(70400, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net6HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net6HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc_c1 = nn.Linear(70400, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(70400, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2


class Net7(nn.Module):
    def __init__(self, num_classes):
        super(Net7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.MaxPool2d(kernel_size=1, stride=2),
        )
        self.fc1 = nn.Linear(21504, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net7HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net7HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.MaxPool2d(kernel_size=1, stride=2),
        )
        self.fc_c1 = nn.Linear(21504, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(21504, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2


class Net8(nn.Module):
    def __init__(self, num_classes):
        super(Net8, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 192, 3, 1),
            nn.Conv2d(192, 320, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(127680, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net8HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net8HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 192, 3, 1),
            nn.Conv2d(192, 320, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_c1 = nn.Linear(127680, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(127680, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2


class Net9(nn.Module):
    def __init__(self, num_classes):
        super(Net9, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.Conv2d(256, 384, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(20736, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net9HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net9HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.Conv2d(256, 384, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_c1 = nn.Linear(20736, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(20736, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2


class Net10(nn.Module):
    def __init__(self, num_classes):
        super(Net10, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.Conv2d(128, 192, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 320, 3, 1),
            nn.Conv2d(320, 448, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc1 = nn.Linear(46592, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net10HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net10HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.Conv2d(128, 192, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 320, 3, 1),
            nn.Conv2d(320, 448, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc_c1 = nn.Linear(46592, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(46592, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2


class Net11(nn.Module):
    def __init__(self, num_classes):
        super(Net11, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.Conv2d(256, 384, 3, 1),
            nn.Conv2d(384, 512, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc1 = nn.Linear(26112, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Net11HT(nn.Module):
    def __init__(self, num_C_cls, num_F_cls):
        super(Net11HT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1),
            nn.Conv2d(256, 384, 3, 1),
            nn.Conv2d(384, 512, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc_c1 = nn.Linear(26112, 512)
        self.dropout_c = nn.Dropout(p=0.5)
        self.fc_c2 = nn.Linear(512, num_C_cls)
        self.fc_f1 = nn.Linear(16128, 512)
        self.dropout_f = nn.Dropout(p=0.5)
        self.fc_f2 = nn.Linear(512, num_F_cls)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x1_d = self.dropout_c(self.fc_c1(x))
        x1 = F.softmax(self.fc_c2(x1_d), dim=1)
        x2_d = self.dropout_f(self.fc_f1(x))
        x2 = F.softmax(self.fc_f2(x2_d), dim=1)

        return x2_d, x1, x2


