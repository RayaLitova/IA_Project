from abc import ABC, abstractmethod
from BaseClasses.Rules import CardGameRules

class GamePhase(ABC):
    def __init__(self, rules : CardGameRules):
        self.rules = rules
        
    @abstractmethod
    def play(self, train : bool = False):
        pass