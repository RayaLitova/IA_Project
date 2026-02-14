from abc import ABC, abstractmethod
from BaseClasses.State import State

class RLTrainReward(ABC):
    @abstractmethod
    def calc_reward(self, state : State, played_idx : int, move) -> float:
        pass
    
class RLTrainRewardFinal(ABC):
    @abstractmethod
    def calc_reward(self, state : State, player_idx : int, player_contributions : list[int], done : bool) -> float:
        pass