from abc import ABC, abstractmethod
import torch.nn as nn
from collections import deque
from BaseClasses.RLAgent import RLAgent
from BaseClasses.RLTrainReward import RLTrainReward, RLTrainRewardFinal
from BaseClasses.Rules import CardGameRules

class RLAgentTrain(ABC):
    def __init__(self, agent : RLAgent, rewards : list[RLTrainReward], final_rewards : list[RLTrainRewardFinal], batch_size : int = 32):
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size
        self.agent = agent
        self.rewards = rewards
        self.final_rewards = final_rewards
    
    @abstractmethod
    def train(self, rules : CardGameRules, episodes : int, save_path : str) -> None:
        pass
    
    def remember(self, new_memory : list) -> None:
        self.memory.append(new_memory)

    @abstractmethod
    def replay(self) -> None:
        pass
