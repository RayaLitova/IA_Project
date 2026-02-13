from BaseClasses.StateEncoder import StateEncoder
import numpy as np
from Belot.BelotRules import BelotRules
from torch import FloatTensor
from BaseClasses.State import State

class BidStateEncoder(StateEncoder):
    def __init__(self):
        # My Hand (32) + CurrentBids (12)
        super().__init__(32 + 12)

    def encode(self, state : State, player_idx : int) -> FloatTensor:
        hand_vec = np.zeros(32)
        for card in state.hands[player_idx]:
            hand_vec[card.id] = 1
        # CurrentBids encoding (12):
        # - first 6 entries: presence (has this contract been bid at least once)
        # - next 6 entries: one-hot of the last non-Pass contract (most recent bidder)
        current_bids_vec = np.zeros(12)
        # presence
        for b in state.played_moves:
            if b != "Pass" and b in BelotRules.CONTRACTS:
                idx = BelotRules.CONTRACTS.index(b)
                current_bids_vec[idx] = 1

        # last non-pass one-hot
        last_non_pass = None
        for b in reversed(state.played_moves):
            if b != "Pass":
                last_non_pass = b
                break
        if last_non_pass and last_non_pass in BelotRules.CONTRACTS:
            current_bids_vec[6 + BelotRules.CONTRACTS.index(last_non_pass)] = 1

        full_vec = np.concatenate([hand_vec, current_bids_vec])
        return FloatTensor(full_vec)