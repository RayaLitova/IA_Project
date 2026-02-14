from BaseClasses.Player import Player
from BaseClasses.State import State
from Belot.BelotRules import BelotRules

class BidPlayer(Player):
    def get_action(self, state : State) -> str:
        print("Your hand (Player 0):", state.hands[0])
        print("Available contracts: 0:AT, 1:NT, 2:♠, 3:♦, 4:♥, 5:♣, 6:Pass")
        try:
            c_idx = int(input("Choose contract: "))
            contract = "Pass" if c_idx == 6 else BelotRules.CONTRACTS[c_idx]
        except:
            contract = 'AT'
        return contract