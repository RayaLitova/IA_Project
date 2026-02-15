from abc import ABC, abstractmethod

class GamePhase(ABC):
    @abstractmethod
    def play(self, train : bool = False):
        pass