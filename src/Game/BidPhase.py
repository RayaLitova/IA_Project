from BidAgent.GamePhase import GamePhase
import time
from BaseClasses.State import State
from Belot.BelotRules import BelotRules
from Belot.Card import RANKS, SUITS, Card
from BidAgent.BidRLAgent import BidRLAgent
from GameAgent.BelotRLAgent import BelotRLAgent
from GameAgent.BelotState import GameState
from BaseClasses.RLAgentPersist import RLAgentPersist
from BidAgent.BidRLAgentTrain import BidRLAgentTrain
from GameAgent.BelotRLAgentTrain import BelotRLAgentTrain
from GameAgent.BelotTrainRewards import BelotTrainFinalRewards, BelotTrainRewards
from BidAgent.BidTrainRewards import BidTrainFinalRewards, BidTrainRewards
from BidAgent.BidDQN import BidDQN
from BidAgent.BidStateEncoder import BidStateEncoder
from GameAgent.BelotDQN import BelotDQN
from GameAgent.BelotStateEncoder import BelotStateEncoder
from Belot.BidPlayer import BidPlayer
from BidAgent.BidAIPlayer import BidAIPlayer
from GameAgent.BelotAIPlayer import BelotAIPlayer
from Belot.BelotPlayer import BelotPlayer
from Belot.BidRules import BidRules

class BidPhase(GamePhase):
    def play(self, train : bool = False):
        belot_agent = BelotRLAgent(BelotDQN(106, 32), BelotStateEncoder())
        RLAgentPersist.load(belot_agent, "models/game/belot_model.pth")
        bid_agent = BidRLAgent(BidDQN(44, 7), BidStateEncoder())
        if train:
            trainer = BidRLAgentTrain(bid_agent, belot_agent, BidTrainRewards, BidTrainFinalRewards)
            trainer.train(20000, "models/bid/bid_model.pth")
        else:
            RLAgentPersist.load(bid_agent, "models/bid/bid_model.pth")
        
        players = [BidPlayer(0), BidAIPlayer(1, bid_agent, train), BidAIPlayer(2, bid_agent, train), BidAIPlayer(3, bid_agent, train)]
        hands = Card.deal_deck()
        bid_state = State(hands, 0, [])
        bid_player_idx = 0
        
        while BelotRules.get_legal_bids(bid_state, BidRules):
            contract = players[bid_player_idx].get_action(bid_state)
            bid_state.played_moves += [contract]
            print(f"Player {bid_player_idx} bid {bid_state.played_moves[-1]}")
            bid_player_idx = (bid_player_idx + 1) % 4
                
        contract = [b for b in bid_state.played_moves if b != "Pass"]
        if not contract:
            return
        print("Final contract:" + contract[-1])
        return hands, contract[-1]