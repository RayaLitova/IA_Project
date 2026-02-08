import copy
from BelotRules import BelotRules

class GameState:
    def __init__(self, contract, hands, played_cards=None, current_trick=None, trick_starter=0, scores=None):
        self.contract = contract
        self.hands = hands 
        self.played_cards = played_cards if played_cards else set()
        self.current_trick = current_trick if current_trick else [] 
        self.trick_starter = trick_starter
        self.scores = scores if scores else [0, 0] # [Team 0 (0&2), Team 1 (1&3)]    

    def get_current_player(self):
        return (self.trick_starter + len(self.current_trick)) % 4

    def is_terminal(self):
        return len(self.hands[0]) == 0 and len(self.current_trick) == 0

    def apply_move(self, card):
        player = self.get_current_player()
        
        new_hands = copy.deepcopy(self.hands)
        if card not in new_hands[player]: # used in random play
             card = new_hands[player][0]
             
        new_hands[player].remove(card)
        
        new_trick = list(self.current_trick) + [card]
        new_played = self.played_cards.copy()
        new_played.add(card)
        
        next_starter = self.trick_starter
        new_scores = list(self.scores)
        rewards = [0, 0]
        
        if len(new_trick) == 4:
            winner_idx, _ = BelotRules.get_trick_winner(next_starter, new_trick, self.contract)
            
            points = sum(BelotRules.get_points(c, self.contract) for c in new_trick)
            
            winning_team = winner_idx % 2
            new_scores[winning_team] += points
            rewards[winning_team] += points 
            
            if len(new_hands[player]) == 0:
                new_scores[winning_team] += 10
                rewards[winning_team] += 10
            
            new_trick = []
            next_starter = winner_idx
        
        new_state = GameState(
            self.contract, 
            new_hands, 
            new_played, 
            new_trick, 
            next_starter, 
            new_scores
        )
        
        return new_state, rewards