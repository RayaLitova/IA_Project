from BaseClasses.PlayRule import PlayRule
from BaseClasses.State import State

class BidRuleRaise(PlayRule):
    def get_legal_moves(self, state : State, hand : list[str]) -> list[str]:
        current_bids = state.played_moves
        bids_count = len(current_bids)
        if bids_count >= state.rules.players_count and current_bids[bids_count - 3:] == ["Pass"] * 3:
            return []
        
        filtered_bids = [b for b in current_bids if b != "Pass"]
        if not filtered_bids: 
            return state.rules.CONTRACTS + ["Pass"]
        
        index = state.rules.CONTRACTS.index(filtered_bids[-1])
        return state.rules.CONTRACTS[:index] + ["Pass"]
    

BidRules = [BidRuleRaise()]

    