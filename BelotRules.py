from Card import ORDER, VALUES, SUITS

class BelotRules:
    @staticmethod
    def get_mode(contract, card_suit):
        if contract == 'AT': return 'AT'
        if contract == 'NT': return 'NT'
        if contract == card_suit: return 'AT'
        return 'NT'

    @staticmethod
    def get_power(card, contract):
        mode = BelotRules.get_mode(contract, card.suit)
        return ORDER[mode].index(card.rank)

    @staticmethod
    def get_points(card, contract):
        mode = BelotRules.get_mode(contract, card.suit)
        return VALUES[mode][card.rank]

    def get_partner(player):
        return (player + 2) % 4
    
    @staticmethod
    def get_trick_winner(trick_starter, trick, contract):
        if not trick: return None, None
        
        lead_card = trick[0]
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
    def get_valid_moves(player, hand, trick_starter, trick, contract):
        if not trick: return hand
        lead_suit = trick[0].suit

        same_suit = [c for c in hand if c.suit == lead_suit]
        if same_suit and (contract == lead_suit or contract == "AT"):
            highest_rank = 0
            for c in trick:
                if VALUES["AT"][c.rank] > highest_rank and c.suit == lead_suit: 
                    highest_rank = VALUES["AT"][c.rank]
            higher = [c for c in same_suit if VALUES["AT"][c.rank] > highest_rank]
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
            highest_trump = max(VALUES["AT"][c.rank], highest_trump)
        trumps = [c for c in hand if c.suit == contract and VALUES["AT"][c.rank] > highest_trump]
        if trumps:
            return trumps

        return hand
