from BaseClasses.Player import Player
from BaseClasses.RLAgent import RLAgent
from BaseClasses.State import State

class BidAIPlayer(Player):
    def __init__(self, index : int, agent : RLAgent, training : bool = False):
        super().__init__(index)
        self.agent = agent
        self.training = training
        
    def get_action(self, state : State) -> str:
        return self.agent.get_action(state, self.index, self.training)