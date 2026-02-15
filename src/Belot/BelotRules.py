from Belot.Card import Card
from BaseClasses.Rules import CardGameRules

class BelotRules(CardGameRules):
    CONTRACTS = ['AT', 'NT', '♠', '♥', '♦', '♣'] # All Trump, No Trump, Suits
    VALUES = {
        'NT': {'7':0, '8':0, '9':0, 'J':2, 'Q':3, 'K':4, '10':10, 'A':11},
        'AT': {'7':0, '8':0, 'Q':3, 'K':4, '10':10, 'A':11, '9':14, 'J':20}
    }
    ORDER = {
        'NT': ['7', '8', '9', 'J', 'Q', 'K', '10', 'A'],
        'AT': ['7', '8', 'Q', 'K', '10', 'A', '9', 'J']
    }
    players_count = 4
    teams_count = 2
    cards_per_player = 8
    
    def get_mode(self, contract : str, card_suit : str) -> str:
        if contract == 'AT': return 'AT'
        if contract == 'NT': return 'NT'
        if contract == card_suit: return 'AT'
        return 'NT'

    def get_power(self, card : Card, contract : str) -> int:
        mode = self.get_mode(contract, card.suit)
        return self.ORDER[mode].index(card.rank)

    def get_points(self, card : Card, contract : str) -> int:
        mode = self.get_mode(contract, card.suit)
        return self.VALUES[mode][card.rank]

    def get_trick_winner(self, state) -> int:
        trick = state.played_moves
        if not trick: return None, None
        contract = state.contract
        trick_starter = state.starting_player
        
        best_play = trick[0]
        winner = trick_starter
        
        for i in range(1, len(trick)):
            curr_card = trick[i]
            curr_player = (trick_starter + i) % self.players_count
            best_card = best_play
            
            is_curr_trump = (contract == 'AT') or (curr_card.suit == contract)
            is_best_trump = (contract == 'AT') or (best_play.suit == contract)
            
            if is_curr_trump and not is_best_trump:
                best_play = curr_card
                winner = curr_player
                continue
            
            if not is_curr_trump and is_best_trump:
                continue
            
            if curr_card.suit == best_card.suit:
                if self.get_power(curr_card, contract) > self.get_power(best_card, contract):
                    best_play = curr_card
                    winner = curr_player
                        
        return (winner, best_play)

    def get_legal_moves(self, state, rules) -> list[Card]:
        player = state.get_current_player()
        hand = state.hands[player]
        if not state.played_moves: 
            return hand
        
        hand_copy = hand.copy()
        for rule in rules:
            hand_copy = rule.get_legal_moves(state, hand_copy)
        if hand_copy:
            return hand_copy
        return hand
    
    def get_legal_bids(self, state, rules) -> list[str]:
        bids = self.CONTRACTS
        for rule in rules:
            bids = rule.get_legal_moves(state, bids)
        return bids