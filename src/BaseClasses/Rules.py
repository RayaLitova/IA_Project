from abc import ABC, abstractmethod
import random
from Belot.Card import RANKS, SUITS, Card

class CardGameRules(ABC):
    players_count = 0
    teams_count = 0
    cards_per_player = 0
    
    def deal_deck(self) -> list[Card]:
        deck = [Card(r, s) for s in SUITS for r in RANKS]
        random.shuffle(deck)
        return {i: sorted(deck[i*self.cards_per_player:(i+1)*self.cards_per_player], key=lambda c: c.id) for i in range(self.players_count)}
    
    @abstractmethod
    def get_mode(self, contract : str, card_suit : str) -> str:
        pass
    
    @abstractmethod
    def get_power(self, card : Card, contract : str) -> int:
        pass

    @abstractmethod
    def get_points(self, card : Card, contract : str) -> int:
        pass
    
    def get_partner(self, player : int) -> int:
        return (player + self.players_count//2) % self.players_count
    
    def get_team(self, player : int) -> int:
        return player % self.teams_count
    
    @abstractmethod
    def get_trick_winner(self, state) -> int:
        pass
    
    @abstractmethod
    def get_legal_moves(self, state, rules) -> list[Card]:
        pass
    
    @abstractmethod
    def get_legal_bids(self, state, rules) -> list[str]:
        pass