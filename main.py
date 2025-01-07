import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from random import sample
from random import seed
import json

from Server import MyServer
from Client import MyClient
import Tools
import ChannelTools
import Model


def Set_Seed(MySeed):
    seed(MySeed)
    torch.manual_seed(MySeed)
    torch.cuda.manual_seed(MySeed)
    torch.cuda.manual_seed_all(MySeed)

def plot_figure():
    pass

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'out_epochs': 30,
        'local_epochs': 1,
        'lr': 0.001,
        'batch_size': 64,
        'device': device,
        'nb_classes': 10,
        'honest_num': 100,
        'select_honest_num': 20,
        'malicious_num': 0,
        'aggregator': 'FedAvg',
        'MySeed': 6
    }

    Set_Seed(config['MySeed'])

    train_dataset = datasets.MNIST(root="D:\\python\\data",
                                   download=True, train=True, transform=transforms.ToTensor())
    each_client_sample_number = int(len(train_dataset) /
                                    (config['honest_num'] + config['malicious_num']))
    train_datasets = random_split(train_dataset,
                                  [each_client_sample_number] * (config['honest_num'] + config['malicious_num']),
                                  generator=torch.Generator().manual_seed(6))

    test_dataset = datasets.MNIST(root="D:\\python\\data",
                                  download=True, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)

    my_Server = MyServer(config, test_loader)
    initial_weights = my_Server.get_parameters()

    Honest_Clients = []
    list_num_sample = []
    Honest_Id = list(range(config['honest_num']))
    for client_num in range(len(Honest_Id)):
        train_loader = DataLoader(dataset=train_datasets[client_num],
                                  batch_size=config['batch_size'],
                                  shuffle=False)
        Honest_Clients.append(MyClient(config, Honest_Id[client_num]+1, train_loader, initial_weights))
        list_num_sample.append(len(train_loader))

    '''
    The all channels need to be generated in the ALTERNATING OPTIMIZATION ALGORITHM 
    '''
    # channels_all, client_transmit_power_all, server_receive_power_all \
    #     = ChannelTools.Alternating_Opti(config['select_honest_num'], config['out_epochs'])

    '''
    The pretraining free model is loading
    '''
    # FreeFCN = Model.MyFCN_Knowledge_Free(config['select_honest_num']).to(device)
    # FreeFCN.load_state_dict(torch.load('free_model_before_train.pth'))
    # FreeFCN.eval()

    list_accuracy = []
    for out_epoch in range(config['out_epochs']):
        print(f"**********The out Epoch [{out_epoch + 1} / {config['out_epochs']}] is as follows: **********")

        list_gradients = []

        # The test of the paper, but it may be unreasonable
        list_avg = []
        list_var = []

        '''
        In the paper, the select_honest_clients in every epoch are non changed. 
        '''
        seed(6)
        select_honest_clients = sample(range(0, 99), config['select_honest_num'])

        # seed(out_epoch)
        # select_honest_clients = sample(range(0, 99), config['select_honest_num'])
        for i in range(config['select_honest_num']):
            client_id = select_honest_clients[i]
            Honest_Clients[client_id].set_client_parameters(my_Server.get_parameters())
            Honest_Clients[client_id].train()

            list_gradients.append(Honest_Clients[client_id].get_gradient())
            list_avg.append(Honest_Clients[client_id].mean_gradient())
            list_var.append(Honest_Clients[client_id].var_gradient())

        gradient = ChannelTools.Error_free(list_gradients, list_num_sample)
        my_Server.update_para(gradient)

        # The test of the paper, but it may be unreasonable
        #list_sk, pi_t = ChannelTools.list_sk_process(list_gradients, list_avg, list_var)

        # error_after_channel = ChannelTools.Full_Power(list_sk, pi_t, config['lr'])
        # error_after_channel = ChannelTools.Channel_Inversion(list_sk, pi_t, config['lr'])
        # error_after_channel = ChannelTools.Alternating_Opti_Error(list_sk, pi_t, config['lr'],
        #                                                           client_transmit_power_all[out_epoch],
        #                                                           server_receive_power_all[out_epoch],
        #                                                           channels_all[out_epoch])
        # error_after_channel = ChannelTools.FCN_Power(list_sk, pi_t, config['lr'], FreeFCN, config['device'])
        # my_Server.update_para(error_after_channel)

        list_accuracy.append(my_Server.evaluate())

    # with open('Error_Free_Accuracy.json', 'w') as file:
    #     json.dump(list_accuracy, file)
    # with open('ErrorFree_accuracy_.json', 'r') as file:
    #     loaded_data = json.load(file)

    print(f"The FL training is over!!!")


if __name__ == '__main__':
    main()
