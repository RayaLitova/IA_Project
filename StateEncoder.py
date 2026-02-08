import numpy as np
from torch import FloatTensor
from Card import CONTRACTS

class StateEncoder:
    def __init__(self):
        # My Hand (32) + Played Cards (32) + Current Trick (32) + Contract (6)
        self.input_size = 32 + 32 + 32 + 6

    def encode(self, state, player_idx):
        hand_vec = np.zeros(32)
        for card in state.hands[player_idx]:
            hand_vec[card.id] = 1
        
        played_vec = np.zeros(32)
        for card in state.played_cards:
            played_vec[card.id] = 1
            
        trick_vec = np.zeros(32)
        for card in state.current_trick:
            trick_vec[card.id] = 1
            
        contract_vec = np.zeros(6)
        if state.contract in CONTRACTS:
            c_idx = CONTRACTS.index(state.contract)
            contract_vec[c_idx] = 1

        full_vec = np.concatenate([hand_vec, played_vec, trick_vec, contract_vec])
        return FloatTensor(full_vec)