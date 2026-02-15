from Belot.Card import Card
from Belot.BelotRules import BelotRules

class State:
    def __init__(self, hands : list[Card], starting_player : int, played_moves : list[Card] = None):
        self.hands = hands 
        self.starting_player = starting_player
        self.played_moves = played_moves
        
    def get_current_player(self) -> int:
        return (self.starting_player + len(self.played_moves)) % BelotRules.players_count