import torch


class Model_one_sigmoid(torch.nn.Module):
    def __init__(self, model_params={'in_size': 250000, 'layers': (100, 10), 'out_size': 1}):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, 1, 1, bias=True)

        self.fc1 = torch.nn.Linear(model_params['in_size'], model_params['layers'][0])
        self.fc2 = torch.nn.Linear(model_params['layers'][0], model_params['layers'][1])
        self.fc3 = torch.nn.Linear(model_params['layers'][1], model_params['out_size'])

    def forward(self, X):
        X = self.conv1(X)
        X = torch.nn.functional.sigmoid(X)
        X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = torch.nn.functional.relu(X)

        X = X.view(-1, 250000)

        X = self.fc1(X)
        X = torch.nn.functional.relu(X)

        X = self.fc2(X)
        X = torch.nn.functional.relu(X)
        X = self.fc3(X)
        X = X.squeeze()


        return X


class Model_with_sigmoid(torch.nn.Module):
    def __init__(self, model_params={'in_size': 250000, 'layers': (100, 10), 'out_size': 1}):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, 1, 1, bias=True)

        self.fc1 = torch.nn.Linear(model_params['in_size'], model_params['layers'][0])
        self.fc2 = torch.nn.Linear(model_params['layers'][0], model_params['layers'][1])
        self.fc3 = torch.nn.Linear(model_params['layers'][1], model_params['out_size'])

    def forward(self, X):
        X = self.conv1(X)
        X = torch.nn.functional.sigmoid(X)
        X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = torch.nn.functional.sigmoid(X)

        X = X.view(-1, 250000)

        X = self.fc1(X)
        X = torch.nn.functional.sigmoid(X)

        X = self.fc2(X)
        X = torch.nn.functional.sigmoid(X)
        X = self.fc3(X)
        X = X.squeeze()


        return X



class StructurelessClf_1(torch.nn.Module):
    def __init__(self, model_params={'in_size': 250000, 'layers': (100, 10), 'out_size': 1}):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, 1, 1, bias=True)

        self.fc1 = torch.nn.Linear(model_params['in_size'], model_params['layers'][0])
        self.fc2 = torch.nn.Linear(model_params['layers'][0], model_params['layers'][1])
        self.fc3 = torch.nn.Linear(model_params['layers'][1], model_params['out_size'])

    def forward(self, X):
        X = self.conv1(X)
        X = torch.nn.functional.relu(X)
        X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv3(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv4(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = X.view(-1, 250000)

        X = self.fc1(X)
        X = torch.nn.functional.relu(X)

        X = self.fc2(X)
        X = torch.nn.functional.relu(X)
        X = self.fc3(X)
        X = X.squeeze()

        # X = torch.nn.functional.sigmoid(X)

        return X

class clf_3d(torch.nn.Module):
    def __init__(self, model_params={'in_size': 250000, 'layers': (100, 10), 'out_size': 1}):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 6, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv3d(6, 16, 3, 1, 1, bias=True)

        self.fc1 = torch.nn.Linear(model_params['in_size'], model_params['layers'][0])
        self.fc2 = torch.nn.Linear(model_params['layers'][0], model_params['layers'][1])
        self.fc3 = torch.nn.Linear(model_params['layers'][1], model_params['out_size'])

    def forward(self, X):
        X = self.conv1(X)
        X = torch.nn.functional.relu(X)
        X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv3(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv4(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = X.view(-1, 250000)

        X = self.fc1(X)
        X = torch.nn.functional.relu(X)

        X = self.fc2(X)
        X = torch.nn.functional.relu(X)
        X = self.fc3(X)
        X = X.squeeze()

        # X = torch.nn.functional.sigmoid(X)

        return X

class StructurelessClf_1_layers(torch.nn.Module):
    def __init__(self, model_params={'in_size': 156250, 'layers': (500, 200), 'out_size': 1}):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(6, 10, 3, 1, 1, bias=True)

        self.fc1 = torch.nn.Linear(model_params['in_size'], model_params['layers'][0])
        self.fc2 = torch.nn.Linear(model_params['layers'][0], model_params['layers'][1])
        self.fc3 = torch.nn.Linear(model_params['layers'][1], model_params['out_size'])

    def forward(self, X):
        X = self.conv1(X)
        X = torch.nn.functional.relu(X)
        X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv3(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv4(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = X.view(-1, 156250)

        X = self.fc1(X)
        X = torch.nn.functional.relu(X)

        X = self.fc2(X)
        X = torch.nn.functional.relu(X)
        X = self.fc3(X)
        X = X.squeeze()

        # X = torch.nn.functional.sigmoid(X)

        return X

class StructurelessClf_1_layers_2(torch.nn.Module):
    def __init__(self, model_params={'in_size': 156250*2, 'layers': (500, 200), 'out_size': 1}):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(10, 20, 3, 1, 1, bias=True)

        self.fc1 = torch.nn.Linear(model_params['in_size'], model_params['layers'][0])
        self.fc2 = torch.nn.Linear(model_params['layers'][0], model_params['layers'][1])
        self.fc3 = torch.nn.Linear(model_params['layers'][1], model_params['out_size'])

    def forward(self, X):
        X = self.conv1(X)
        X = torch.nn.functional.relu(X)
        X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv3(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv4(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = X.view(-1, 156250*2)

        X = self.fc1(X)
        X = torch.nn.functional.relu(X)

        X = self.fc2(X)
        X = torch.nn.functional.relu(X)
        X = self.fc3(X)
        X = X.squeeze()

        # X = torch.nn.functional.sigmoid(X)

        return X


class StructurelessClf_1_layers_3(torch.nn.Module):
    def __init__(self, model_params={'in_size': 156250*2, 'layers': (500, 200, 10), 'out_size': 1}):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(10, 20, 3, 1, 1, bias=True)

        self.fc1 = torch.nn.Linear(model_params['in_size'], model_params['layers'][0])
        self.fc2 = torch.nn.Linear(model_params['layers'][0], model_params['layers'][1])
        self.fc3 = torch.nn.Linear(model_params['layers'][1], model_params['layers'][2])
        self.fc4 = torch.nn.Linear(model_params['layers'][2], model_params['out_size'])

    def forward(self, X):
        X = self.conv1(X)
        X = torch.nn.functional.relu(X)
        X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv3(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv4(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = X.view(-1, 156250*2)

        X = self.fc1(X)
        X = torch.nn.functional.relu(X)

        X = self.fc2(X)
        X = torch.nn.functional.relu(X)

        X = self.fc3(X)
        X = torch.nn.functional.relu(X)

        X = self.fc4(X)
        X = X.squeeze()

        # X = torch.nn.functional.sigmoid(X)

        return X


class StructurelessClf_2(torch.nn.Module):
    def __init__(self, model_params={'in_size': 156250*2, 'layers': (500, 200, 10), 'out_size': 1}):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(6, 10, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(10, 20, 3, 1, 1, bias=True)

        self.fc1 = torch.nn.Linear(model_params['in_size'], model_params['layers'][0])
        self.fc2 = torch.nn.Linear(model_params['layers'][0], model_params['layers'][1])
        self.fc3 = torch.nn.Linear(model_params['layers'][1], model_params['layers'][2])
        self.fc4 = torch.nn.Linear(model_params['layers'][2], model_params['out_size'])

    def forward(self, X):
        X = self.conv1(X)
        X = torch.nn.functional.relu(X)
        X = torch.nn.functional.avg_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = self.conv3(X)
        X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        # X = self.conv4(X)
        # X = torch.nn.functional.relu(X)
        # X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = X.view(-1, 156250*2)

        X = self.fc1(X)
        X = torch.nn.functional.relu(X)

        X = self.fc2(X)
        X = torch.nn.functional.relu(X)

        X = self.fc3(X)
        X = torch.nn.functional.relu(X)

        X = self.fc4(X)
        X = X.squeeze()

        # X = torch.nn.functional.sigmoid(X)

        return X