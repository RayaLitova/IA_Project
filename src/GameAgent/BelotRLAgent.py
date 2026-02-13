import copy
import random
import numpy as np
import torch
from BaseClasses.RLAgent import RLAgent
from Belot.BelotRules import BelotRules
from Belot.Card import CONTRACTS, ORDER, RANKS, SUITS, Card
from GameAgent.BelotStateEncoder import BelotStateEncoder
from GameAgent.BelotDQN import BelotDQN
from GameAgent.BelotState import GameState
from BaseClasses.State import State

class BelotRLAgent(RLAgent):
    def __init__(self):
        super().__init__(BelotDQN(106, 32), BelotStateEncoder(), epsilon = 0.1)
        
    def get_action(self, state : State, player_idx : int, training : bool = False) -> Card:
        legal_moves = BelotRules.get_valid_moves(player_idx, state.hands[player_idx], state.starting_player, state.played_moves, state.contract)
        
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)

        state_tensor = self.encoder.encode(state, player_idx)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        legal_ids = [c.id for c in legal_moves]
        mask = torch.full(q_values.shape, -float('inf'))
        mask[legal_ids] = 0
        
        masked_q_values = q_values + mask
        best_card_id = torch.argmax(masked_q_values).item()
        
        for c in legal_moves:
            if c.id == best_card_id:
                return c
        return legal_moves[0]

    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = []
        targets = []
        
        for state, action_id, reward, next_state, done, pid in batch:
            state_tensor = self.encoder.encode(state, pid)
            target = reward
            if not done:
                next_state_tensor = self.encoder.encode(next_state, pid)
                with torch.no_grad():
                    target = reward + 0.95 * torch.max(self.model(next_state_tensor)).item()
            
            current_qs = self.model(state_tensor).detach().clone()
            current_qs[action_id] = target
            
            states.append(state_tensor.numpy())
            targets.append(current_qs.numpy())

        input_batch = torch.FloatTensor(np.array(states))
        target_batch = torch.FloatTensor(np.array(targets))
        
        self.optimizer.zero_grad()
        outputs = self.model(input_batch)
        loss = self.criterion(outputs, target_batch)
        loss.backward()
        self.optimizer.step()
        
    def train(self, episodes : int, save_path : str) -> None:
        print(f"Training Neural Network for {episodes} games...")
        print("Team Setup: Player 0 (You) & Player 2 (Friendly AI) vs Player 1 & Player 3 (Opponents)")
        
        deck = [Card(r, s) for s in SUITS for r in RANKS]
        
        team0_wins = 0
        team1_wins = 0
        recent_scores = []
        
        for e in range(episodes):
            random.shuffle(deck)
            hands = {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(4)}
            contract = random.choice(CONTRACTS)
            
            state = GameState(contract, hands)
            game_experiences = []
            
            player_contributions = {i: 0 for i in range(4)}
            
            while not state.is_terminal():
                pid = state.get_current_player()
                curr_state_obj = copy.deepcopy(state)
                
                card = self.get_action(state, pid, training=True)
                next_state, trick_rewards = state.apply_move(card)
                
                player_contributions[pid] += trick_rewards[pid % 2]
                
                game_experiences.append({
                    'state': curr_state_obj,
                    'action_id': card.id,
                    'trick_rewards': trick_rewards,
                    'next_state': next_state,
                    'done': next_state.is_terminal(),
                    'player_idx': pid
                })
                
                state = next_state
            
            final_scores = state.scores
            recent_scores.append((final_scores[0], final_scores[1]))
            if len(recent_scores) > 100:
                recent_scores.pop(0)
            
            if final_scores[0] > final_scores[1]:
                game_bonus = [100, -100]
                team0_wins += 1
            elif final_scores[1] > final_scores[0]:
                game_bonus = [-100, 100]
                team1_wins += 1
            else:
                game_bonus = [0, 0]
            
            score_diff = final_scores[0] - final_scores[1]
            
            for exp in game_experiences:
                pid = exp['player_idx']
                team = pid % 2
                partner_idx = (pid + 2) % 4
                
                reward = exp['trick_rewards'][team] * 0.5
                
                card_played = [c for c in exp['state'].hands[pid] if c.id == exp['action_id']][0]
                
                
                if len(exp['state'].played_moves) >= 1:
                    current_winner_idx, _ = BelotRules.get_trick_winner(
                        exp['state'].starting_player,
                        exp['state'].played_moves, 
                        exp['state'].contract
                    )
                    
                    if current_winner_idx == partner_idx:
                        card_value = BelotRules.get_points(card_played, exp['state'].contract)
                        if card_value >= 10:
                            reward += 15
                        else:
                            reward += 5
                    
                    elif current_winner_idx != pid:
                        card_value = BelotRules.get_points(card_played, exp['state'].contract)
                        
                        temp_trick = exp['state'].played_moves + [card_played]
                        new_winner_idx, _ = BelotRules.get_trick_winner(
                            exp['state'].starting_player,
                            temp_trick,
                            exp['state'].contract
                        )
                        
                        if new_winner_idx == pid or new_winner_idx == partner_idx:
                            reward += 20
                        elif card_value >= 10:
                            reward -= 25
                        else:
                            reward += 3
                
                is_trump = (exp['state'].contract == "AT" or exp['state'].contract == card_played.suit)
                order = ORDER["AT"] if is_trump else ORDER["NT"]
                
                if card_played.rank == order[-1]:
                    second_highest = Card(rank=order[-2], suit=card_played.suit)
                    has_protection = second_highest in exp['state'].hands[pid]
                    
                    temp_trick = exp['state'].played_moves + [card_played]
                    winner_idx, _ = BelotRules.get_trick_winner(
                        exp['state'].starting_player,
                        temp_trick,
                        exp['state'].contract
                    )
                    
                    if winner_idx != pid and winner_idx != partner_idx and not has_protection:
                        reward -= 40
                    elif len(exp['state'].played_moves) == 0 and not has_protection:
                        reward -= 30
                    elif has_protection:
                        reward += 8
                
                lead_suit = exp['state'].played_moves[0].suit if exp['state'].played_moves else None
                if lead_suit and card_played.suit != lead_suit:
                    has_lead_suit = any(c.suit == lead_suit for c in exp['state'].hands[pid])
                    if has_lead_suit:
                        reward -= -float('inf')
                
                if card_played.suit == exp['state'].contract:
                    if len(exp['state'].played_moves) >= 1:
                        current_winner_idx, _ = BelotRules.get_trick_winner(
                            exp['state'].starting_player,
                            exp['state'].played_moves,
                            exp['state'].contract
                        )
                        if current_winner_idx == partner_idx:
                            reward -= 20
                
                if exp['done']:
                    reward += game_bonus[team]
                    
                    if team == 0:
                        reward += score_diff * 0.3
                    else:
                        reward -= score_diff * 0.3
                    
                    contribution_bonus = player_contributions[pid] * 0.2
                    reward += contribution_bonus
                
                self.remember((
                    exp['state'],
                    exp['action_id'],
                    reward,
                    exp['next_state'],
                    exp['done'],
                    pid
                ))
            
            replay_times = 4 if e < 1000 else 8 if e < 5000 else 16
            for _ in range(replay_times):
                self.replay()
            
            if e % 50 == 0:
                self.epsilon = max(0.05, self.epsilon * 0.95)
            
            if e % 100 == 0:
                avg_team0 = sum(s[0] for s in recent_scores) / len(recent_scores) if recent_scores else 0
                avg_team1 = sum(s[1] for s in recent_scores) / len(recent_scores) if recent_scores else 0
                win_rate = team0_wins / (e + 1) * 100
                
                print(f"Episode {e}/{episodes} | "
                    f"Score: T0={final_scores[0]} T1={final_scores[1]} | "
                    f"Avg(100): T0={avg_team0:.1f} T1={avg_team1:.1f} | "
                    f"Win Rate: {win_rate:.1f}% | "
                    f"Epsilon: {self.epsilon:.3f}")
            
            if (e + 1) % 1000 == 0:
                self.save(save_path, e+1)
                print(f"Model saved at episode {e+1}")

        print("\n=== Training Complete ===")
        print(f"Team 0 Wins: {team0_wins} ({team0_wins/episodes*100:.1f}%)")
        print(f"Team 1 Wins: {team1_wins} ({team1_wins/episodes*100:.1f}%)")
        self.save(save_path, episodes)
