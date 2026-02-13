import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from BaseClasses.State import State
from BaseClasses.DQN import DQN
from BaseClasses.StateEncoder import StateEncoder

class RLAgent:
    def __init__(self, model : DQN, encoder : StateEncoder, epsilon=0.1, batch_size = 32):
        self.model = model
        self.encoder = encoder
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size

    def get_action(self, state : State, player_idx : int, training = False):
        pass
        
    def remember(self, new_memory : list) -> None:
        self.memory.append(new_memory)

    def replay(self) -> None:
        pass
    
    def save(self, filepath : str, episode : int) -> None:
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'memory_size': len(self.memory)
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath : str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found!")
        
        checkpoint = torch.load(filepath)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {filepath} (epsilon={checkpoint.get('epsilon', 0.1):.3f})")
    
    def train(self, episodes : int, save_path : str) -> None:
        pass