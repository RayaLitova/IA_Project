import random
import torch
from BaseClasses.RLAgent import RLAgent
from Belot.Card import Card
from BaseClasses.State import State
from BaseClasses.DQN import DQN
from BaseClasses.StateEncoder import StateEncoder
from Belot.BelotPlayRules import BelotPlayRules

class BelotRLAgent(RLAgent):
    def __init__(self, model : DQN, encoder : StateEncoder):
        super().__init__(model, encoder, epsilon = 0.1)
        
    def get_action(self, state : State, player_idx : int, training : bool = False) -> Card:
        legal_moves = state.rules.get_legal_moves(state, BelotPlayRules)
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