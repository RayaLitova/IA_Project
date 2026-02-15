from abc import ABC, abstractmethod
from BaseClasses.State import State

class PlayRule(ABC): 
    @abstractmethod
    def get_legal_moves(self, state : State, hand):
        pass