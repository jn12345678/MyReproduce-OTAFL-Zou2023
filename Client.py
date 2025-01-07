import torch
import Model


class MyClient:
    def __init__(self, config, client_id, dataloader, initial_weights):
        self.id = client_id
        self.train_dataloader = dataloader

        self.local_epoch = config['local_epochs']
        self.device = config['device']
        self.nb_classes = config['nb_classes']
        self.lr = config['lr']

        self.model = Model.MyCNN1(self.nb_classes)
        if initial_weights is not None:
            self.set_client_parameters(initial_weights)

        self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.gradient = None
        self.parameter_len = None

    def get_id(self):
        return self.id

    def get_num_sample(self):
        return len(self.train_dataloader)

    def get_gradient(self):
        return self.gradient

    def set_client_parameters(self, initial_weights):
        for j, param in enumerate(self.model.parameters()):
            param.data = initial_weights[j].data.clone().detach()

    def get_client_parameters(self):
        return [param.clone().detach() for param in self.model.parameters()]

    def calculate_gradient(self, parameters_before):
        return [torch.sub(a, b) for a, b in zip(parameters_before, self.get_client_parameters())]

    def train(self):
        parameters_before = self.get_client_parameters()

        for epoch in range(self.local_epoch):
            #print(f"The train Epoch [{epoch + 1} / {self.local_epoch}] in the client {self.id} is starting!")
            for data, targets in self.train_dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                scores = self.model(data)
                loss = self.criterion(scores, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.gradient = self.calculate_gradient(parameters_before)

    def gradient_normalize(self):
        # Some problems lead to non convergence
        tmp = len(self.gradient)-1
        avg = self.gradient[tmp].sum()
        tmp2 = len(self.gradient[tmp])
        for i in range(tmp):
            avg += self.gradient[i].sum()
            tmp2 += len(self.gradient[i])
        avg /= tmp2
        var = ((self.gradient[tmp] - avg) ** 2).sum()
        for i in range(tmp):
            var += ((self.gradient[i] - avg) ** 2).sum()
        var /= tmp2

        return [(gradient_tmp - avg) / var.sqrt() for gradient_tmp in self.gradient]

    def gradient_layer_normalize(self):
        # Some problems lead to non convergence
        return [(gradient_tmp - gradient_tmp.mean()) / gradient_tmp.std()
                for gradient_tmp in self.gradient]

    def mean_gradient(self):
        tmp = torch.tensor(0.0).to(self.device)
        self.parameter_len = torch.tensor(0).to(self.device)
        for gradient_tmp in self.gradient:
            tmp += gradient_tmp.sum()
            self.parameter_len += len(gradient_tmp)
        return tmp / self.parameter_len

    def var_gradient(self):
        mean = self.mean_gradient()
        tmp = torch.tensor(0.0).to(self.device)
        for gradient_tmp in self.gradient:
            tmp += ((gradient_tmp - mean) ** 2).sum()

        return tmp / self.parameter_len
