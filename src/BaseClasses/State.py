from Belot.Card import Card
from BaseClasses.Rules import CardGameRules

class State:
    def __init__(self, rules : CardGameRules, hands : list[Card], starting_player : int, played_moves : list[Card] = None):
        self.hands = hands 
        self.starting_player = starting_player
        self.played_moves = played_moves if played_moves else [] 
        self.rules = rules
        
    def get_current_player(self) -> int:
        return (self.starting_player + len(self.played_moves)) % self.rules.players_count