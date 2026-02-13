from torch import FloatTensor
from BaseClasses.State import State
from abc import ABC, abstractmethod

class StateEncoder(ABC):
    def __init__(self, input_size : int):
        self.input_size = input_size

    @abstractmethod
    def encode(self, state : State, player_idx : int) -> FloatTensor:
        pass