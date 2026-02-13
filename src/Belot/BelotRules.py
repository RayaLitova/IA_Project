from Belot.Card import Card

class BelotRules:
    CONTRACTS = ['AT', 'NT', '♠', '♥', '♦', '♣'] # All Trump, No Trump, Suits
    VALUES = {
        'NT': {'7':0, '8':0, '9':0, 'J':2, 'Q':3, 'K':4, '10':10, 'A':11},
        'AT': {'7':0, '8':0, 'Q':3, 'K':4, '10':10, 'A':11, '9':14, 'J':20}
    }
    ORDER = {
        'NT': ['7', '8', '9', 'J', 'Q', 'K', '10', 'A'],
        'AT': ['7', '8', 'Q', 'K', '10', 'A', '9', 'J']
    }
    
    @staticmethod
    def get_mode(contract : str, card_suit : str) -> str:
        if contract == 'AT': return 'AT'
        if contract == 'NT': return 'NT'
        if contract == card_suit: return 'AT'
        return 'NT'

    @staticmethod
    def get_power(card : Card, contract : str) -> int:
        mode = BelotRules.get_mode(contract, card.suit)
        return BelotRules.ORDER[mode].index(card.rank)

    @staticmethod
    def get_points(card : Card, contract : str) -> int:
        mode = BelotRules.get_mode(contract, card.suit)
        return BelotRules.VALUES[mode][card.rank]

    def get_partner(player : int) -> int:
        return (player + 2) % 4
    
    @staticmethod
    def get_trick_winner(trick_starter : int, trick : list[Card], contract : str) -> int:
        if not trick: return None, None
        
        best_play = trick[0]
        winner = trick_starter
        
        for i in range(1, len(trick)):
            curr_card = trick[i]
            curr_player = (trick_starter + i) % 4
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
                if BelotRules.get_power(curr_card, contract) > BelotRules.get_power(best_card, contract):
                    best_play = curr_card
                    winner = curr_player
                        
        return (winner, best_play)

    @staticmethod
    def get_valid_moves(player : int, hand : list[Card], trick_starter : int, trick : list[Card], contract : str) -> list[Card]:
        if not trick: return hand
        lead_suit = trick[0].suit

        same_suit = [c for c in hand if c.suit == lead_suit]
        if same_suit and (contract == lead_suit or contract == "AT"):
            highest_rank = 0
            for c in trick:
                if BelotRules.get_power(c, contract) > highest_rank and c.suit == lead_suit: 
                    highest_rank = BelotRules.get_power(c, contract)
            higher = [c for c in same_suit if BelotRules.get_power(c, contract) > highest_rank]
            if higher: 
                return higher
            return same_suit 
        elif same_suit:
            return same_suit

        (winner, _) = BelotRules.get_trick_winner(trick_starter, trick, contract)
        if winner == BelotRules.get_partner(player) or contract == "AT" or contract == "NT":
            return hand

        played_trumps = [c for c in trick if c.suit == contract]
        highest_trump = 0
        for c in played_trumps:
            highest_trump = max(BelotRules.get_power(c, contract), highest_trump)
        trumps = [c for c in hand if c.suit == contract and BelotRules.get_power(c, contract) > highest_trump]
        if trumps:
            return trumps

        return hand

    def get_legal_bids(current_bids : list[str]) -> list[str]:
        bids_count = len(current_bids)
        if bids_count >= 4 and current_bids[bids_count - 3:] == ["Pass"] * 3:
            return []
        
        filtered_bids = [b for b in current_bids if b != "Pass"]
        if not filtered_bids: 
            return BelotRules.CONTRACTS + ["Pass"]
        
        index = BelotRules.CONTRACTS.index(filtered_bids[-1])
        return BelotRules.CONTRACTS[:index] + ["Pass"]