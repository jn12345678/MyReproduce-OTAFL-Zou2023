import torch
import torch.nn as nn
import torch.nn.functional as F


class MyCNN1(nn.Module):
    """
    a convolutional neural network (CNN) with two convolution
    layers and two fully-connected layers for train MNIST
    """

    def __init__(self, channels=1, num_class=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8,
                               kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyFCN_Knowledge_Free(nn.Module):
    """
        a fully connected network (FCN) without knowledge for train MNIST
    """
    def __init__(self, client_num):
        super().__init__()
        self.fc1 = nn.Linear(client_num, 256)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, client_num+1)
        self.BN1 = nn.BatchNorm1d(256)
        self.BN2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.BN1(x))
        x = self.fc2(x)
        x = F.relu(self.BN2(x))
        x = self.output(x)
        return x


class Structure_Map(nn.Module):
    def __init__(self, num_clients, P_MAX):
        super(Structure_Map, self).__init__()
        self.num_clients = num_clients
        self.P_MAX = P_MAX

    def forward(self, multiplier, channels):
        batch_size = len(channels)
        tmp = torch.empty(batch_size, self.num_clients+1).to(channels.device)
        for i in range(batch_size):
            tmp1 = torch.mul(multiplier[i][self.num_clients], channels[i])
            tmp2 = channels[i] + torch.mul(multiplier[i, 0:self.num_clients], multiplier[i][self.num_clients])
            tmp[i,0:self.num_clients] = torch.min(torch.div(tmp1, torch.pow(tmp2, 2)), self.P_MAX)
        tmp[:,self.num_clients] = multiplier[:, self.num_clients]

        return tmp


class Structure_Map_Back(torch.autograd.Function):
    """
    num_clients = 20
    P_MAX = 30.0
    """
    def forward(self, multiplier, channels):
        self.save_for_backward = [multiplier, channels]

        num_clients = 20
        P_MAX = torch.tensor([30.0]).to(channels.device)
        batch_size = len(channels)
        tmp = torch.empty(batch_size, num_clients+1).to(channels.device)
        for i in range(batch_size):
            tmp1 = torch.mul(multiplier[i][num_clients], channels[i])
            tmp2 = channels[i] + torch.mul(multiplier[i, 0:num_clients], multiplier[i][num_clients])
            tmp[i,0:num_clients] = torch.min(torch.div(tmp1, torch.pow(tmp2, 2)), P_MAX)
        tmp[:,num_clients] = multiplier[:, num_clients]

        return tmp

    def backward(self, grad_out):
        multiplier, channels = self.save_for_backward

        return grad_out


class MyFCN_Knowledge_Guided(nn.Module):
    """
        a fully connected network (FCN) with knowledge for train MNIST
    """

    def __init__(self, client_num, P_MAX):
        super().__init__()
        self.fc1 = nn.Linear(client_num, 256)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, client_num + 1)
        self.BN1 = nn.BatchNorm1d(256)
        self.BN2 = nn.BatchNorm1d(64)
        self.BN3 = nn.BatchNorm1d(21)
        self.Sg = nn.Sigmoid()
        self.Structure_Map = Structure_Map(client_num, P_MAX)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        y = self.fc1(x)
        y = F.relu(self.BN1(y))
        y = self.fc2(y)
        y = F.relu(self.BN2(y))
        y = self.output(y)
        y = self.BN3(y)
        y = self.Sg(y)
        return self.Structure_Map(y, x)