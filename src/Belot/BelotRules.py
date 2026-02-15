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

    @staticmethod
    def get_partner(player : int) -> int:
        return (player + 2) % 4
    
    @staticmethod
    def get_trick_winner(state) -> int:
        trick = state.played_moves
        if not trick: return None, None
        contract = state.contract
        trick_starter = state.starting_player
        
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
    def get_legal_moves(state, rules) -> list[Card]:
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

    def get_legal_bids(state, rules) -> list[str]:
        bids = BelotRules.CONTRACTS
        for rule in rules:
            bids = rule.get_legal_moves(state, bids)
        return bids