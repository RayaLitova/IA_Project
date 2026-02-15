from BaseClasses.GamePhase import GamePhase
import time
from Belot.Card import Card
from GameAgent.BelotRLAgent import BelotRLAgent
from GameAgent.BelotState import GameState
from BaseClasses.RLAgentPersist import RLAgentPersist
from GameAgent.BelotRLAgentTrain import BelotRLAgentTrain
from GameAgent.BelotTrainRewards import BelotTrainFinalRewards, BelotTrainRewards
from GameAgent.BelotDQN import BelotDQN
from GameAgent.BelotStateEncoder import BelotStateEncoder
from GameAgent.BelotAIPlayer import BelotAIPlayer
from Belot.BelotPlayer import BelotPlayer
from BaseClasses.Rules import CardGameRules


class BelotPhase(GamePhase):
    def __init__(self, rules : CardGameRules, hands = None, contract = None):
        super().__init__(rules)
        self.hands = hands
        self.contract = contract
        
    def play(self, train : bool = False) -> None:
        hands = self.hands
        contract = self.contract
        agent = BelotRLAgent(BelotDQN(106, 32), BelotStateEncoder())
        players = [BelotPlayer(self.rules, 0), BelotAIPlayer(self.rules, 1, agent, train), BelotAIPlayer(self.rules, 2, agent, train), BelotAIPlayer(self.rules, 3, agent, train)]
        
        if train:
            trainer = BelotRLAgentTrain(agent, BelotTrainRewards, BelotTrainFinalRewards)
            trainer.train(self.rules, 20000, "models/game/belot_model.pth")
        else:
            RLAgentPersist.load(agent, "models/game/belot_model.pth")
        
        if not hands:   
            hands = self.rules.deal_deck()
            
        if not contract:
            print("Your hand (Player 0):", hands[0])
            print("Available contracts: 0:AT, 1:NT, 2:♠, 3:♦, 4:♥, 5:♣")
            try:
                c_idx = int(input("Choose contract: "))
                contract = self.rules.CONTRACTS[c_idx]
            except:
                contract = 'AT'
        
        state = GameState(self.rules, contract, hands)
        
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