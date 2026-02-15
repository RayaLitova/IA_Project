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
from Game.BelotPhase import BelotPhase
from Game.BidPhase import BidPhase

class Game:
    def start(self):
        print("\n1. Train new model")
        print("2. Load model and play without bid")
        print("3. Train bid model")
        print("4. Load model and play with bid")
        
        choice = input("\nYour choice (1/2/3/4): ").strip()
        
        if choice == "1":
            self.play(True)
        elif choice == "2":
            try:
                self.play()
            except FileNotFoundError as e:
                print(e)
                print("Train a model first!")
        elif choice == "3":
            try:
                self.play_with_bid(True)
            except FileNotFoundError as e:
                print(e)
                print("Train a belot model first!")
        elif choice == "4":
            try:
                self.play_with_bid()
            except FileNotFoundError as e:
                print(e)
                print("Train a model first!")
        else:
            print("Invalid choice. Run the script again.")
            
    def play(self, train = False):
        BelotPhase().play(train)
    
    def play_with_bid(self, train = False):
        hands, contract = BidPhase().play(train)
        BelotPhase(hands, contract).play()
            
        