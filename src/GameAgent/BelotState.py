import copy
from BaseClasses.State import State
from Belot.BelotRules import BelotRules
from Belot.Card import Card

class GameState(State):
    def __init__(self, contract : str, hands : list[Card], played_cards : list[Card] = None, played_moves : list[Card] = None, starting_player : int = 0, scores : list[int] = None):
        self.contract = contract
        self.hands = hands 
        self.played_cards = played_cards if played_cards else set()
        self.played_moves = played_moves if played_moves else [] 
        self.starting_player = starting_player
        self.scores = scores if scores else [0, 0] # [Team 0 (0&2), Team 1 (1&3)]    

    def is_terminal(self) -> bool:
        return len(self.hands[0]) == 0 and len(self.played_moves) == 0

    def apply_move(self, card : Card) -> tuple[State, list[int]]:
        player = self.get_current_player()
        
        new_hands = copy.deepcopy(self.hands)
        if card not in new_hands[player]: # used in random play
             card = new_hands[player][0]
             
        new_hands[player].remove(card)
        
        new_trick = list(self.played_moves) + [card]
        new_played = self.played_cards.copy()
        new_played.add(card)
        
        next_starter = self.starting_player
        new_scores = list(self.scores)
        rewards = [0, 0]
        
        new_state = GameState(
            self.contract, 
            new_hands, 
            new_played, 
            new_trick, 
            next_starter, 
            new_scores
        )
        
        if len(new_trick) == BelotRules.players_count:
            winner_idx, _ = BelotRules.get_trick_winner(new_state)
            points = sum(BelotRules.get_points(c, self.contract) for c in new_trick)
            winning_team = BelotRules.get_team(winner_idx)
            new_scores[winning_team] += points
            rewards[winning_team] += points 
            
            if len(new_hands[player]) == 0:
                new_scores[winning_team] += 10
                rewards[winning_team] += 10
            
            new_trick = []
        
        new_state = GameState(
            self.contract, 
            new_hands, 
            new_played, 
            new_trick, 
            next_starter, 
            new_scores
        )
        
        return new_state, rewards