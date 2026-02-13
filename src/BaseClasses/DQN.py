import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size : int, initial_out : int, output_size : int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, initial_out)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(initial_out, initial_out//2)
        self.fc3 = nn.Linear(initial_out//2, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)