import random
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
            
    def play(self, train : bool = False, hands : list[Card] = None, contract : str = None) -> None:
        agent = BelotRLAgent(BelotDQN(106, 32), BelotStateEncoder())
        players = [BelotPlayer(0), BelotAIPlayer(1, agent, train), BelotAIPlayer(2, agent, train), BelotAIPlayer(3, agent, train)]
        
        if train:
            trainer = BelotRLAgentTrain(agent, BelotTrainRewards, BelotTrainFinalRewards)
            trainer.train(20000, "models/game/belot_model.pth")
        else:
            RLAgentPersist.load(agent, "models/game/belot_model.pth")
        
        # todo
        if not hands:   
            deck = [Card(r, s) for s in SUITS for r in RANKS]
            random.shuffle(deck)
            hands = {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(4)}
            
        if not contract:
            print("Your hand (Player 0):", hands[0])
            print("Available contracts: 0:AT, 1:NT, 2:♠, 3:♦, 4:♥, 5:♣")
            try:
                c_idx = int(input("Choose contract: "))
                contract = BelotRules.CONTRACTS[c_idx]
            except:
                contract = 'AT'
        
        state = GameState(contract, hands)
        
        print(f"\nContract: {contract}")
        print("You (Player 0) + Friendly AI (Player 2)")
        print("Opponent AI (Player 1) + Opponent AI (Player 3)")
        
        while not state.is_terminal():
            pid = state.get_current_player()
            card = players[pid].get_action(state)
            print(f"Player {pid} plays: {card}")
            time.sleep(0.3)
            state, _ = state.apply_move(card)
            
            if len(state.played_moves) == 0 and not state.is_terminal():
                print(f"Trick complete! Scores: Your Team {state.scores[0]} - {state.scores[1]} Opponent Team")
        
        print(f"\nGAME OVER! Final Score: Your Team {state.scores[0]} - {state.scores[1]} Opponent Team")
        if state.scores[0] > state.scores[1]:
            print("You WIN!")
        elif state.scores[1] > state.scores[0]:
            print("You LOSE!")
        else:
            print("It's a DRAW!")
        
    def play_with_bid(self, train : bool = False) -> None:
        belot_agent = BelotRLAgent(BelotDQN(106, 32), BelotStateEncoder())
        RLAgentPersist.load(belot_agent, "models/game/belot_model.pth")
        bid_agent = BidRLAgent(BidDQN(44, 7), BidStateEncoder())
        if train:
            trainer = BidRLAgentTrain(bid_agent, belot_agent, BidTrainRewards, BidTrainFinalRewards)
            trainer.train(20000, "models/bid/bid_model.pth")
        else:
            RLAgentPersist.load(bid_agent, "models/bid/bid_model.pth")
        
        players = [BidPlayer(0), BidAIPlayer(1, bid_agent, train), BidAIPlayer(2, bid_agent, train), BidAIPlayer(3, bid_agent, train)]
        deck = [Card(r, s) for s in SUITS for r in RANKS]
        random.shuffle(deck)
        hands = {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(4)}
        bid_state = State(hands, 0, [])
        bid_player_idx = 0
        
        while BelotRules.get_legal_bids(bid_state.played_moves):
            contract = players[bid_player_idx].get_action(bid_state)
            bid_state.played_moves += [contract]
            print(f"Player {bid_player_idx} bid {bid_state.played_moves[-1]}")
            bid_player_idx = (bid_player_idx + 1) % 4
                
        contract = [b for b in bid_state.played_moves if b != "Pass"]
        if not contract:
            return
        print("Final contract:" + contract[-1])
        self.play(False, hands, contract[-1])
            
        