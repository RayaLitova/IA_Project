import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from BelotRules import BelotRules
from collections import deque

class RLAgent:
    def __init__(self, model, encoder, epsilon=0.1):
        self.model = model
        self.encoder = encoder
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

    def get_action(self, state, player_idx, training=False):
        legal_moves = BelotRules.get_valid_moves(player_idx, state.hands[player_idx], state.trick_starter, state.current_trick, state.contract)
        
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)

        state_tensor = self.encoder.encode(state, player_idx)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        legal_ids = [c.id for c in legal_moves]
        mask = torch.full(q_values.shape, -float('inf'))
        mask[legal_ids] = 0
        
        masked_q_values = q_values + mask
        best_card_id = torch.argmax(masked_q_values).item()
        
        for c in legal_moves:
            if c.id == best_card_id:
                return c
        return legal_moves[0]

    def remember(self, state, action_id, reward, next_state, done, player_idx):
        self.memory.append((state, action_id, reward, next_state, done, player_idx))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = []
        targets = []
        
        for state, action_id, reward, next_state, done, pid in batch:
            state_tensor = self.encoder.encode(state, pid)
            
            target = reward
            if not done:
                next_state_tensor = self.encoder.encode(next_state, pid)
                with torch.no_grad():
                    target = reward + 0.95 * torch.max(self.model(next_state_tensor)).item()
            
            current_qs = self.model(state_tensor).detach().clone()
            current_qs[action_id] = target
            
            states.append(state_tensor.numpy())
            targets.append(current_qs.numpy())

        input_batch = torch.FloatTensor(np.array(states))
        target_batch = torch.FloatTensor(np.array(targets))
        
        self.optimizer.zero_grad()
        outputs = self.model(input_batch)
        loss = self.criterion(outputs, target_batch)
        loss.backward()
        self.optimizer.step()