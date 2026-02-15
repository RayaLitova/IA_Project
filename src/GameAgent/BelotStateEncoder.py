import numpy as np
from torch import FloatTensor
from BaseClasses.StateEncoder import StateEncoder
from Belot.BelotRules import BelotRules
from BaseClasses.State import State

class BelotStateEncoder(StateEncoder):
    def __init__(self):
        # My Hand (32) + Played Cards (32) + Current Trick (32) + Contract (6) + Winner (players count)
        super().__init__(32 + 32 + 32 + 6 + BelotRules.players_count)

    def encode(self, state : State, player_idx : int) -> FloatTensor:
        hand_vec = np.zeros(32)
        for card in state.hands[player_idx]:
            hand_vec[card.id] = 1
        
        played_vec = np.zeros(32)
        for card in state.played_cards:
            played_vec[card.id] = 1
            
        trick_vec = np.zeros(32)
        for card in state.played_moves:
            trick_vec[card.id] = 1
            
        contract_vec = np.zeros(6)
        if state.contract in BelotRules.CONTRACTS:
            c_idx = BelotRules.CONTRACTS.index(state.contract)
            contract_vec[c_idx] = 1

        winner_vec = np.zeros(BelotRules.players_count)  # One-hot: which player is winning
        if state.played_moves:
            winner_idx, _ = BelotRules.get_trick_winner(state)
            if winner_idx is not None:
                winner_vec[winner_idx] = 1

        full_vec = np.concatenate([hand_vec, played_vec, trick_vec, contract_vec, winner_vec])
        return FloatTensor(full_vec)