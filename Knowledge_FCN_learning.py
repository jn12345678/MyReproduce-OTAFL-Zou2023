import Model
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from random import seed

import ChannelTools


class MyChannelDataset(Dataset):
    def __init__(self, features, labels):
        self.train_features = features
        self.train_labels = labels

    def __len__(self):
        return len(self.train_features)

    def __getitem__(self, index):
        feature = torch.tensor(self.train_features[index], dtype=torch.float32)
        label = torch.tensor(self.train_labels[index], dtype=torch.float32)

        return feature, label


class My_MSE_Regularize(nn.Module):
    def __init__(self):
        super(My_MSE_Regularize, self).__init__()
        self.regularize = 0.01
        self.P_BAR = 10.0

    def forward(self, channels, powers):
        #powers = torch.abs(powers)
        batch_size = len(channels)
        index = len(channels[0])
        loss = 0.0
        for i in range(batch_size):
            tmp = torch.div(torch.mul(channels[i], powers[i][0:index]), powers[i][index])
            tmp = torch.pow(torch.sqrt(tmp) - 1.0, 2)
            loss += torch.sum(tmp)
            loss += torch.div(1.0, powers[i][index])
        loss /= batch_size
        loss += self.regularize * torch.sum(F.relu(torch.div(torch.sum(powers[0:batch_size], 0), batch_size)
                                                   - self.P_BAR))

        return loss


def pretraining_FCN(Net, Loss_function, Epochs, Device, Train_dataloader):
    print(f"The pretraining of FCN is starting!")
    loss_dic = []
    for epoch in range(Epochs):
        # print(f"The train Epoch [{epoch + 1} / {epochs}] of pretraining FCN is starting!")
        epoch_loss = 0.0
        for data, targets in Train_dataloader:
            data, targets = data.to(Device), targets.to(Device)
            output = Net(data)
            loss = Loss_function(data, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        loss_dic.append(epoch_loss)
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, epoch_loss))


def Set_Seed(MySeed):
    seed(MySeed)
    torch.manual_seed(MySeed)
    torch.cuda.manual_seed(MySeed)
    torch.cuda.manual_seed_all(MySeed)


num_clients = 20
training_size = 5000
#test_size = 100
epochs = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Set_Seed(6)

train_channels_all, train_client_transmit_power_all, train_server_receive_power_all \
    = ChannelTools.Generated_Channel_For_FCN(num_clients, training_size)
train_features = []
train_labels = train_client_transmit_power_all
for t in range(len(train_channels_all)):
    train_features.append([(channel.real ** 2 + channel.imag ** 2) for channel in train_channels_all[t]])
    train_labels[t].append(train_server_receive_power_all[t])

MyChannelData = MyChannelDataset(train_features, train_labels)
train_dataloader = DataLoader(MyChannelData, batch_size=1000, shuffle=False)

# MyFCN = Model.MyFCN_Knowledge_Free(num_clients).to(device)
# loss_function = My_MSE_Regularize_Free().to(device)

P_MAX = torch.tensor(30.0).to(device)
MyFCN = Model.MyFCN_Knowledge_Guided(num_clients, P_MAX).to(device)

loss_function = My_MSE_Regularize().to(device)

optimizer = torch.optim.Adam(MyFCN.parameters(), 0.01)
pretraining_FCN(MyFCN, loss_function, epochs, device, train_dataloader)
path = 'Guided_model_after_train.pth'
torch.save(MyFCN.state_dict(), path)

