class State:
    def __init__(self, hands, starting_player, played_moves = None):
        self.hands = hands 
        self.starting_player = starting_player
        self.played_moves = played_moves
        
    def get_current_player(self):
        return (self.starting_player + len(self.played_moves)) % 4