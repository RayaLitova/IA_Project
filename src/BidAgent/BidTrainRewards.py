from BaseClasses.State import State
from BaseClasses.RLTrainReward import RLTrainRewardFinal

class BidFinalScoresReward(RLTrainRewardFinal):
    def calc_reward(self, state : State, player_idx : int) -> float:
        final_scores = state.scores
        team = state.rules.get_team(player_idx)
        point_diff = final_scores[team] - final_scores[1 - team]
        if point_diff > 0:
            base_reward = 100
            margin_bonus = min(point_diff / 5, 50)
            return base_reward + margin_bonus
        elif point_diff < 0:
            base_reward = -100
            margin_penalty = max(point_diff / 5, -50)
            return base_reward + margin_penalty
        else:
            return 0
        
BidTrainRewards = []
BidTrainFinalRewards = [BidFinalScoresReward()]