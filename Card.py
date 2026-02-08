SUITS = ['♠', '♦', '♥', '♣'] # Spades, Diamonds, Hearts, Clubs
RANKS = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CONTRACTS = ['AT', 'NT', '♠', '♦', '♥', '♣'] # All Trump, No Trump, Suits

CARD_TO_ID = {f"{r}{s}": i for i, (s, r) in enumerate([(s, r) for s in SUITS for r in RANKS])}
ID_TO_CARD = {v: k for k, v in CARD_TO_ID.items()}

VALUES = {
    'NT': {'7':0, '8':0, '9':0, 'J':2, 'Q':3, 'K':4, '10':10, 'A':11},
    'AT': {'7':0, '8':0, 'Q':3, 'K':4, '10':10, 'A':11, '9':14, 'J':20}
}
ORDER = {
    'NT': ['7', '8', '9', 'J', 'Q', 'K', '10', 'A'],
    'AT': ['7', '8', 'Q', 'K', '10', 'A', '9', 'J']
}

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.str_rep = f"{rank}{suit}"
        self.id = CARD_TO_ID[self.str_rep]

    def __repr__(self): return self.str_rep
    def __eq__(self, other): return self.id == other.id
    def __hash__(self): return self.id