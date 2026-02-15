from BaseClasses.Player import Player
from BaseClasses.State import State
from Belot.BelotRules import BelotRules
from Belot.Card import Card
from Belot.BelotPlayRules import BelotPlayRules

class BelotPlayer(Player):
    def get_action(self, state : State) -> Card:
        legal = BelotRules.get_legal_moves(state, BelotPlayRules)
        print(f"\nTable: {state.played_moves if state.played_moves else 'Empty'}")
        print(f"Your Hand: {[f'{i}:{c}' for i,c in enumerate(legal)]}")
        idx = int(input("Choose card index: "))
        return legal[idx]