# This is the tool to process the channel

import torch
from math import sqrt
from random import gauss
from random import seed

import Tools


# The energy of noise
# SIGMA = 1

# SNR = P_BAR / SIGMA^2
# SNR = 10

# The constraints of transmit
# P_BAR = 10 ** (SNR / 10) * SIGMA = 10
# P_MAX = 3 * P_BAR = 30


def Error_free(list_gradients, list_num_sample):
    return Tools.FedAvg(list_gradients, list_num_sample)


def list_sk_process(list_gradients, list_avg, list_var):
    avg_mean = torch.tensor(list_avg).float().mean()
    avg_var = torch.tensor(list_var).float().mean()
    pi_t = avg_var.sqrt()
    list_sk = []
    for i in range(len(list_gradients)):
        list_sk.append([(gradient_tmp - avg_mean) / pi_t for gradient_tmp in list_gradients[i]])
    return list_sk, pi_t


def Error_in_trans(client_transmit_power,
                   server_receive_power,
                   channels,
                   list_sk,
                   pi_t,
                   lr,
                   SIGMA=1,
                   device='cuda'):
    client_num = len(list_sk)
    s_t = list_sk[client_num - 1]
    Error_trans = [tmp
                   * sqrt(client_transmit_power[client_num - 1] / server_receive_power
                          * (channels[client_num - 1].real ** 2
                             + channels[client_num - 1].imag ** 2))
                   + SIGMA / sqrt(server_receive_power) * torch.randn(tmp.shape).to(device)
                   for tmp in s_t]
    for j in range(len(s_t)):
        for i in range(client_num - 1):
            Error_trans[j] += sqrt(client_transmit_power[i] / server_receive_power
                                   * (channels[i].real ** 2 + channels[i].imag ** 2)) * list_sk[i][j]
            s_t[j] += list_sk[i][j]

    for i in range(len(Error_trans)):
        Error_trans[i] -= s_t[i]
        Error_trans[i] *= (lr * pi_t / len(list_sk))

    return Error_trans


def Full_Power(list_sk, pi_t, lr, SIGMA=1.0, P_BAR=10.0):
    seed(6)
    channels = [complex(gauss(0, 1), gauss(0, 1)) for _ in list_sk]
    client_transmit_power = [P_BAR] * len(list_sk)
    tmp1 = 0.0
    tmp2 = 0.0
    for channel in channels:
        tmp = channel.real ** 2 + channel.imag ** 2
        tmp1 += tmp
        tmp2 += sqrt(tmp)
    server_receive_power = ((SIGMA ** 2 + P_BAR * tmp1) / tmp2) ** 2 / P_BAR

    Error_trans = Error_in_trans(client_transmit_power,
                                 server_receive_power,
                                 channels,
                                 list_sk,
                                 pi_t,
                                 lr,
                                 SIGMA=1)

    return Error_trans


def Channel_Inversion(list_sk, pi_t, lr, SIGMA=1.0, P_BAR=10.0, epsilon_c=0.1):
    seed(6)
    channels = [complex(gauss(0, 1), gauss(0, 1)) for _ in list_sk]
    tmp1 = []
    tmp2 = []
    for i in range(len(channels)):
        tmp1.append(channels[i].real ** 2 + channels[i].imag ** 2)
        tmp2.append((SIGMA ** 2 + P_BAR * tmp1[i]) / sqrt(P_BAR * tmp1[i]))
    server_receive_power = min(tmp2)
    client_transmit_power = []
    for i in range(len(list_sk)):
        client_transmit_power.append(min(P_BAR, server_receive_power / tmp1[i])
                                     if (P_BAR * tmp1[i]) >= epsilon_c else 0)
    Error_trans = Error_in_trans(client_transmit_power,
                                 server_receive_power,
                                 channels,
                                 list_sk,
                                 pi_t,
                                 lr,
                                 SIGMA=1)

    return Error_trans


def calculate_eta_t(channels_power_t,
                    transmit_power_t,
                    SIGMA_t):
    tmp1 = SIGMA_t ** 2
    tmp2 = 0.0
    for k in range(len(channels_power_t)):
        tmp = transmit_power_t[k] * (channels_power_t[k] ** 2)
        tmp1 += tmp
        tmp2 += sqrt(tmp)
    server_receive_power_t = (tmp1 / tmp2) ** 2

    return server_receive_power_t


def calculate_trans_power_k_bisection(k,
                                      channels_power_all,
                                      server_receive_power,
                                      P_BAR,
                                      P_MAX):
    mu_min = 0.0
    mu_max = 1.0
    mu = mu_min
    epsilon = 1e-6

    out_epoch = len(channels_power_all)
    power_k_all = [0.0] * out_epoch
    iter_max = 1000
    index = 0
    while index < iter_max:
        index += 1
        # print(f"===== This is the {index} iteration of Bisection! =====")
        for t in range(out_epoch):
            eta_t = server_receive_power[t]
            power_k_all[t] = min((eta_t * channels_power_all[t][k])
                                 / (channels_power_all[t][k] + mu * eta_t) ** 2,
                                 P_MAX)

        if abs(sum(power_k_all) - out_epoch * P_BAR) < epsilon:
            break
        elif sum(power_k_all) > out_epoch * P_BAR:
            mu_min = mu
            mu = (mu + mu_max) / 2
        else:
            mu_max = mu
            mu = (mu + mu_min) / 2

    return power_k_all


def calculate_MSE_bar(channels_power_all,
                      client_transmit_power_all,
                      server_receive_power_all,
                      SIGMA):
    mse = 0.0
    for t in range(len(channels_power_all)):
        for k in range(len(channels_power_all[0])):
            mse += (sqrt(client_transmit_power_all[t][k] *
                         channels_power_all[t][k] / server_receive_power_all[t]) - 1.0) ** 2
        mse += SIGMA ** 2 / server_receive_power_all[t]

    return mse


def Alternating_Opti(clients_num,
                     epoch_num,
                     epsilon=1e-6,
                     SIGMA=1,
                     P_BAR=10.0,
                     P_MAX=30.0):
    iter_max = 100
    channels_all = []
    channels_power_all = []
    client_transmit_power_all = []
    seed(6)
    for _ in range(epoch_num):
        tmp = [complex(gauss(0, 1), gauss(0, 1)) for _ in range(clients_num)]
        channels_all.append(tmp)
        channels_power_all.append([(channel.real ** 2 + channel.imag ** 2) for channel in tmp])
        client_transmit_power_all.append([P_BAR] * clients_num)

    server_receive_power_all = [1.0] * epoch_num

    mse_before = 0.0
    for index in range(iter_max):
        print(f"##### This is the {index} iteration of Alternating_Optimization! #####")
        condition_1 = [0.0] * clients_num
        for t in range(epoch_num):
            server_receive_power_all[t] = calculate_eta_t(channels_power_all[t],
                                                          client_transmit_power_all[t],
                                                          SIGMA)
            for k in range(clients_num):
                client_transmit_power_all[t][k] = min(server_receive_power_all[t] / channels_power_all[t][k], P_MAX)
                condition_1[k] += client_transmit_power_all[t][k]

        for k in range(clients_num):
            if condition_1[k] > (epoch_num * P_BAR):
                power_adjust_k = calculate_trans_power_k_bisection(k,
                                                                   channels_power_all,
                                                                   server_receive_power_all,
                                                                   P_BAR,
                                                                   P_MAX)
                for t in range(epoch_num):
                    client_transmit_power_all[t][k] = power_adjust_k[t]

        mse_later = calculate_MSE_bar(channels_power_all,
                                      client_transmit_power_all,
                                      server_receive_power_all,
                                      SIGMA)
        if (abs(mse_before - mse_later) / mse_later) < epsilon:
            break
        else:
            mse_before = mse_later

    return channels_all, client_transmit_power_all, server_receive_power_all


def Alternating_Opti_Error(list_sk,
                           pi_t,
                           lr,
                           client_transmit_power,
                           server_receive_power,
                           channels):
    return Error_in_trans(client_transmit_power,
                          server_receive_power,
                          channels,
                          list_sk,
                          pi_t,
                          lr)


def Generated_Channel_For_FCN(clients_num,
                              epoch_num,
                              P_BAR=10.0):
    channels_all = []
    server_receive_power_all = [1.0] * epoch_num
    client_transmit_power_all = []
    seed(6)
    for _ in range(epoch_num):
        tmp = [complex(gauss(0, 1), gauss(0, 1)) for _ in range(clients_num)]
        channels_all.append(tmp)
        client_transmit_power_all.append([P_BAR] * clients_num)

    return channels_all, client_transmit_power_all, server_receive_power_all


def FCN_Power(list_sk, pi_t, lr, FreeFCN, device):
    seed(8)
    channels = [complex(gauss(0, 1), gauss(0, 1)) for _ in list_sk]
    features = torch.tensor([channel.real ** 2 + channel.imag ** 2 for channel in channels]).to(device)
    features = features[None, :]
    tmp = torch.abs(FreeFCN(features))
    server_receive_power = tmp[0][len(list_sk)].tolist()
    client_transmit_power = tmp[0][0:len(list_sk)].tolist()
    Error_trans = Error_in_trans(client_transmit_power,
                                 server_receive_power,
                                 channels,
                                 list_sk,
                                 pi_t,
                                 lr,
                                 SIGMA=1)

    return Error_trans

