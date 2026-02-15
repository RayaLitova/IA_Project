from Belot.BelotRules import BelotRules
from BaseClasses.PlayRule import PlayRule
from Belot.Card import Card
from GameAgent.BelotState import GameState

class PlayRuleFollowSuit(PlayRule):
    def get_legal_moves(self, state : GameState, hand : list[Card]) -> list[Card]:
        lead_suit = state.played_moves[0].suit
        same_suit = [c for c in hand if c.suit == lead_suit]
        if same_suit and (state.contract == lead_suit or state.contract == "AT"):
            highest_rank = 0
            for c in state.played_moves:
                if BelotRules.get_power(c, state.contract) > highest_rank and c.suit == lead_suit: 
                    highest_rank = BelotRules.get_power(c, state.contract)
            higher = [c for c in same_suit if BelotRules.get_power(c, state.contract) > highest_rank]
            if higher: 
                return higher
            return same_suit 
        elif same_suit:
            return same_suit
        return hand
    

class PlayRuleRaise(PlayRule):
    def get_legal_moves(self, state : GameState, hand : list[Card]) -> list[Card]:
        player = state.get_current_player()
        (winner, _) = BelotRules.get_trick_winner(state)
        if winner == BelotRules.get_partner(player) or state.contract == "AT" or state.contract == "NT":
            return hand

        played_trumps = [c for c in state.played_moves if c.suit == state.contract]
        highest_trump = 0
        for c in played_trumps:
            highest_trump = max(BelotRules.get_power(c, state.contract), highest_trump)
        trumps = [c for c in hand if c.suit == state.contract and BelotRules.get_power(c, state.contract) > highest_trump]
        if trumps:
            return trumps
        
        return hand
    
BelotPlayRules = [PlayRuleFollowSuit(), PlayRuleRaise()]

    