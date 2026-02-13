from BaseClasses.State import State
from BaseClasses.DQN import DQN
from BaseClasses.StateEncoder import StateEncoder
from abc import ABC, abstractmethod
import torch.optim as optim

class RLAgent(ABC):
    def __init__(self, model : DQN, encoder : StateEncoder, epsilon=0.1, batch_size = 32):
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.model = model
        self.encoder = encoder
        self.epsilon = epsilon
        
    @abstractmethod
    def get_action(self, state : State, player_idx : int, training = False):
        pass