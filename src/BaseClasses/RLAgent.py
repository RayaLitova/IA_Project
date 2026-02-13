import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class RLAgent:
    def __init__(self, model, encoder, epsilon=0.1, batch_size = 32):
        self.model = model
        self.encoder = encoder
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size

    def get_action(self, state, player_idx, training=False):
        pass
        
    def remember(self, new_memory):
        self.memory.append(new_memory)

    def replay(self):
        pass
    
    def save(self, filepath, episode):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'memory_size': len(self.memory)
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found!")
        
        checkpoint = torch.load(filepath)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {filepath} (epsilon={checkpoint.get('epsilon', 0.1):.3f})")
    
    def train(self, episodes, save_path):
        pass