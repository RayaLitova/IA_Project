from abc import ABC, abstractmethod
from Belot.Card import Card
from BaseClasses.State import State

class PlayRule(ABC):
    @abstractmethod
    def get_legal_moves(self, state : State, hand : list[Card]) -> list[Card]:
        pass