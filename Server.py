import torch
import Model


class MyServer:
    def __init__(self, config, dataloader):
        self.model = Model.MyCNN1()
        self.model = self.model.to(config['device'])

        self.device = config['device']

        self.test_dataloader = dataloader

    def set_parameters(self, list_Tensor):
        for j, param in enumerate(self.model.parameters()):
            param.data = list_Tensor[j].data.clone().detach()

    def get_parameters(self):
        return [param.clone().detach() for param in self.model.parameters()]

    def evaluate(self):
        total_num = 0
        correct_num = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_num += targets.size(0)
                correct_num += (predicted == targets).sum().item()

            accuracy = 100 * (float(correct_num) / float(total_num))
            print(f"==========The accuracy of test data on the server is {accuracy:.2f}%")

        return accuracy

    def update_para(self, gradients):
        for j, param in enumerate(self.model.parameters()):
            param.data = torch.sub(param.clone().detach(), gradients[j])
