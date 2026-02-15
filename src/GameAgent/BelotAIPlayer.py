from BaseClasses.Player import Player
from BaseClasses.State import State
from Belot.Card import Card
from BaseClasses.RLAgent import RLAgent
from BaseClasses.Rules import CardGameRules

class BelotAIPlayer(Player):
    def __init__(self, rules : CardGameRules, index : int, agent : RLAgent, training : bool = False):
        super().__init__(rules, index)
        self.agent = agent
        self.training = training
         
    def get_action(self, state : State) -> Card:
        return self.agent.get_action(state, self.index, self.training)