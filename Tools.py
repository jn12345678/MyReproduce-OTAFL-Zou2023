# This is the tool for FL.
import torch


def FedAvg(list_gradients, list_num_sample):
    length = len(list_gradients)
    all_num_sample = sum(list_num_sample)
    avg_gradient = list_gradients[length - 1]
    for j in range(len(avg_gradient)):
        avg_gradient[j] *= list_num_sample[length - 1]
        avg_gradient[j] /= all_num_sample
        for i in range(length - 1):
            avg_gradient[j] = torch.add(avg_gradient[j],
                                        list_gradients[i][j] * list_num_sample[i] / all_num_sample)

    return avg_gradient


def Geometric_Median(list_gradients, list_num_sample):
    #iteration_num = 1
    nu = 1e-6

    length = len(list_gradients)
    alpha = []
    for i in range(length):
        alpha.append(list_num_sample[i] / length)

    geo_gradient = list_gradients[length-1]
    for j in range(len(list_gradients[0])):
        for i in range(length-1):
            # beta_i = alpha[i] / max(nu, torch.norm(list_gradients[i][j], p=2))
            pass

