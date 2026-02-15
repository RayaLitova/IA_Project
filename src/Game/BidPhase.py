from BaseClasses.GamePhase import GamePhase
from BaseClasses.State import State
from Belot.Card import Card
from BidAgent.BidRLAgent import BidRLAgent
from GameAgent.BelotRLAgent import BelotRLAgent
from BaseClasses.RLAgentPersist import RLAgentPersist
from BidAgent.BidRLAgentTrain import BidRLAgentTrain
from BidAgent.BidTrainRewards import BidTrainFinalRewards, BidTrainRewards
from BidAgent.BidDQN import BidDQN
from BidAgent.BidStateEncoder import BidStateEncoder
from GameAgent.BelotDQN import BelotDQN
from GameAgent.BelotStateEncoder import BelotStateEncoder
from Belot.BidPlayer import BidPlayer
from BidAgent.BidAIPlayer import BidAIPlayer
from Belot.BidRules import BidRules

class BidPhase(GamePhase):
    def play(self, train : bool = False):
        belot_agent = BelotRLAgent(BelotDQN(106, 32), BelotStateEncoder())
        RLAgentPersist.load(belot_agent, "models/game/belot_model.pth")
        bid_agent = BidRLAgent(BidDQN(44, 7), BidStateEncoder())
        if train:
            trainer = BidRLAgentTrain(bid_agent, belot_agent, BidTrainRewards, BidTrainFinalRewards)
            trainer.train(self.rules, 20000, "models/bid/bid_model.pth")
        else:
            RLAgentPersist.load(bid_agent, "models/bid/bid_model.pth")
        
        players = [BidPlayer(self.rules, 0), BidAIPlayer(self.rules, 1, bid_agent, train), BidAIPlayer(self.rules, 2, bid_agent, train), BidAIPlayer(self.rules, 3, bid_agent, train)]
        hands = self.rules.deal_deck()
        bid_state = State(self.rules, hands, 0, [])
        bid_player_idx = 0
        
        while self.rules.get_legal_bids(bid_state, BidRules):
            contract = players[bid_player_idx].get_action(bid_state)
            bid_state.played_moves += [contract]
            print(f"Player {bid_player_idx} bid {bid_state.played_moves[-1]}")
            bid_player_idx = (bid_player_idx + 1) % self.rules.players_count
                
        contract = [b for b in bid_state.played_moves if b != "Pass"]
        if not contract:
            return None, None
        print("Final contract:" + contract[-1])
        return hands, contract[-1]