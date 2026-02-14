from abc import ABC, abstractmethod
from BaseClasses.State import State

class Player(ABC):
    def __init__(self, index : int):
        self.index = index
        
    @abstractmethod
    def get_action(state : State):
        pass