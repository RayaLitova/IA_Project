from torch import FloatTensor
from BaseClasses.State import State

class StateEncoder:
    def __init__(self, input_size : int):
        self.input_size = input_size

    def encode(self, state : State, player_idx : int) -> FloatTensor:
        pass