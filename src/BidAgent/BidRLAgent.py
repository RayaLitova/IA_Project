import random
import torch
from BaseClasses.RLAgent import RLAgent
from BaseClasses.State import State 
from Belot.BelotRules import BelotRules
from BaseClasses.DQN import DQN
from BaseClasses.StateEncoder import StateEncoder

class BidRLAgent(RLAgent):
    def __init__(self, model : DQN, encoder : StateEncoder):
        super().__init__(model, encoder, epsilon = 0.9)
        
    def get_action(self, state : State, player_idx : int, training : bool = False) -> str:
        legal_bids = BelotRules.get_legal_bids(state.played_moves)
        if not legal_bids: 
            return
        if training and random.random() < self.epsilon:
            return random.choice(legal_bids)

        state_tensor = self.encoder.encode(state, player_idx)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        legal_ids = [6 if c == "Pass" else BelotRules.CONTRACTS.index(c) for c in legal_bids]
        mask = torch.full_like(q_values, -float('inf'))
        mask[legal_ids] = 0
        
        masked_q_values = q_values + mask
        best_bid_id = torch.argmax(masked_q_values).item()
        
        for b in legal_ids:
            if b == best_bid_id:
                return "Pass" if b == 6 else BelotRules.CONTRACTS[b]
        return legal_bids[0]
