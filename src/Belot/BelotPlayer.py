from BaseClasses.Player import Player
from BaseClasses.State import State
from Belot.BelotRules import BelotRules
from Belot.Card import Card

class BelotPlayer(Player):
    def get_action(self, state : State) -> Card:
        legal = BelotRules.get_valid_moves(self.index, state.hands[0], state.starting_player, state.played_moves, state.contract)
        print(f"\nTable: {state.played_moves if state.played_moves else 'Empty'}")
        print(f"Your Hand: {[f'{i}:{c}' for i,c in enumerate(legal)]}")
        idx = int(input("Choose card index: "))
        return legal[idx]