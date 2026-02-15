import copy
from BaseClasses.State import State
from BaseClasses.RLTrainReward import RLTrainReward, RLTrainRewardFinal
from Belot.Card import Card

class FinalScoresReward(RLTrainRewardFinal):
    def calc_reward(self, state : State, player_idx : int) -> float:
        final_scores = state.scores
        reward = 0
        team = state.rules.get_team(player_idx)
        
        if final_scores[0] > final_scores[1]:
            game_bonus = [100, -100]
        elif final_scores[1] > final_scores[0]:
            game_bonus = [-100, 100]
        else:
            game_bonus = [0, 0]
        
        score_diff = final_scores[0] - final_scores[1]
        
        reward += game_bonus[team]
        if team == 0:
            reward += score_diff * 0.3
        else:
            reward -= score_diff * 0.3            
        return reward
        
    
class PartnerWinningReward(RLTrainReward):
    def calc_reward(self, state : State, player_idx : int, played_card) -> float:
        partner_idx = state.rules.get_partner(player_idx)
        current_winner_idx, _ = state.rules.get_trick_winner(state)
        
        if current_winner_idx == partner_idx:
            card_value = state.rules.get_points(played_card, state.contract)
            if card_value >= 10:
                return 15
            else:
                return 5
        return 0
    
class OpponentWinningReward(RLTrainReward):
    def calc_reward(self, state : State, player_idx : int, played_card) -> float:
        current_winner_idx, _ = state.rules.get_trick_winner(state)
        if current_winner_idx == player_idx:
            return 0
        partner_idx = state.rules.get_partner(player_idx)
        card_value = state.rules.get_points(played_card, state.contract)
        
        temp_state = copy.copy(state)
        temp_state.played_moves = state.played_moves + [played_card]
        new_winner_idx, _ = state.rules.get_trick_winner(temp_state)
        
        if new_winner_idx == player_idx or new_winner_idx == partner_idx:
            return 20
        elif card_value >= 10:
            return -25
        else:
            return 3

class HighCardProtectionReward(RLTrainReward):
    def calc_reward(self, state : State, player_idx : int, played_card) -> float:
        is_trump = (state.contract == "AT" or state.contract == played_card.suit)
        order = state.rules.ORDER["AT"] if is_trump else state.rules.ORDER["NT"]
        
        if played_card.rank != order[-1]:
            return 0
        
        second_highest = Card(rank=order[-2], suit=played_card.suit)
        has_protection = second_highest in state.hands[player_idx]
        partner_idx = state.rules.get_partner(player_idx)
        
        temp_state = copy.copy(state)
        temp_state.played_moves = state.played_moves + [played_card]
        winner_idx, _ = state.rules.get_trick_winner(temp_state)
        
        if winner_idx != player_idx and winner_idx != partner_idx and not has_protection:
            return -40
        elif len(state.played_moves) == 0 and not has_protection:
            return -30
        elif has_protection:
            return 8
        else:
            return 0

class ValidMoveReward(RLTrainReward):
    def calc_reward(self, state : State, player_idx : int, played_card) -> float:
        lead_suit = state.played_moves[0].suit if state.played_moves else None
        if lead_suit and played_card.suit != lead_suit:
            has_lead_suit = any(c.suit == lead_suit for c in state.hands[player_idx])
            if has_lead_suit:
                return -float('inf')
        return 0
    
class TrumpingPartnerReward(RLTrainReward):
    def calc_reward(self, state : State, player_idx : int, played_card) -> float:
        if played_card.suit != state.contract:
            return 0
        if len(state.played_moves) < 1:
            return 0
        
        current_winner_idx, _ = state.rules.get_trick_winner(state)
        partner_idx = state.rules.get_partner(player_idx)
        if current_winner_idx == partner_idx:
            return -20
        return 0

BelotTrainRewards = [PartnerWinningReward(), 
                     TrumpingPartnerReward(), 
                     ValidMoveReward(), 
                     HighCardProtectionReward(),
                     OpponentWinningReward(),
                     PartnerWinningReward()]

BelotTrainFinalRewards = [FinalScoresReward()]