from abc import ABC, abstractmethod
from BaseClasses.State import State
from BaseClasses.Rules import CardGameRules

class Player(ABC):
    def __init__(self, rules : CardGameRules, index : int):
        self.index = index
        self.rules = rules
        
    @abstractmethod
    def get_action(state : State):
        pass