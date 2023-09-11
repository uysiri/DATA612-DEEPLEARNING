import torch
from torch import nn

# create class for model
# this one is going to have more layers

class MultipleRegressionDeep(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegressionDeep, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 14)
        self.layer_2 = nn.Linear(14, 28)
        self.layer_3 = nn.Linear(28, 28)
        self.layer_4 = nn.Linear(28, 28)
        self.layer_5 = nn.Linear(28, 28)
        self.layer_out = nn.Linear(28, 1)
        
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.relu(self.layer_5(x))
        x = self.layer_out(x)
        return (x)
    def predict(self, test_inputs):
        x = self.relu(self.layer_1(test_inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.relu(self.layer_5(x))
        x = self.layer_out(x)
        return (x)