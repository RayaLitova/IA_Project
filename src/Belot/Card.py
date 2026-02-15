from __future__ import annotations
import random

SUITS = ['♠', '♥', '♦', '♣'] # Spades, Hearts, Diamonds, Clubs
RANKS = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CARD_TO_ID = {f"{r}{s}": i for i, (s, r) in enumerate([(s, r) for s in SUITS for r in RANKS])}
ID_TO_CARD = {v: k for k, v in CARD_TO_ID.items()}

class Card:
    def __init__(self, rank : str, suit : str):
        self.rank = rank
        self.suit = suit
        self.str_rep = f"{rank}{suit}"
        self.id = CARD_TO_ID[self.str_rep]

    @staticmethod
    def deal_deck(players_count : int) -> list[Card]:
        deck = [Card(r, s) for s in SUITS for r in RANKS]
        random.shuffle(deck)
        return {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(players_count)}
        
    def __repr__(self): return self.str_rep
    def __eq__(self, other): return self.id == other.id
    def __hash__(self): return self.id