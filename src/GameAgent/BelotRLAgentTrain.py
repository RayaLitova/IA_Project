import copy
import numpy as np
from Belot.Card import RANKS, SUITS, Card
import random
import torch
from Belot.BelotRules import BelotRules
from Belot.Card import Card
from GameAgent.BelotState import GameState
from BaseClasses.RLAgentPersist import RLAgentPersist
from BaseClasses.RLAgentTrain import RLAgentTrain

class BelotRLAgentTrain(RLAgentTrain):
    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = []
        targets = []
        
        for state, action_id, reward, next_state, done, pid in batch:
            state_tensor = self.agent.encoder.encode(state, pid)
            target = reward
            if not done:
                next_state_tensor = self.agent.encoder.encode(next_state, pid)
                with torch.no_grad():
                    target = reward + 0.95 * torch.max(self.agent.model(next_state_tensor)).item()
            
            current_qs = self.agent.model(state_tensor).detach().clone()
            current_qs[action_id] = target
            
            states.append(state_tensor.numpy())
            targets.append(current_qs.numpy())

        input_batch = torch.FloatTensor(np.array(states))
        target_batch = torch.FloatTensor(np.array(targets))
        
        self.agent.optimizer.zero_grad()
        outputs = self.agent.model(input_batch)
        loss = self.criterion(outputs, target_batch)
        loss.backward()
        self.agent.optimizer.step()
        
    def train(self, episodes : int, save_path : str) -> None:
        print(f"Training Neural Network for {episodes} games...")
        print("Team Setup: Player 0 (You) & Player 2 (Friendly AI) vs Player 1 & Player 3 (Opponents)")
        
        deck = [Card(r, s) for s in SUITS for r in RANKS]
  
        recent_scores = []
        
        for e in range(episodes):
            random.shuffle(deck)
            hands = {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(4)}
            contract = random.choice(BelotRules.CONTRACTS)
            
            state = GameState(contract, hands)
            game_experiences = []
            
            player_contributions = {i: 0 for i in range(4)}
            
            while not state.is_terminal():
                pid = state.get_current_player()
                curr_state_obj = copy.deepcopy(state)
                
                card = self.agent.get_action(state, pid, training=True)
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
            
            for exp in game_experiences:
                pid = exp['player_idx']
                team = pid % 2
                reward = exp['trick_rewards'][team] * 0.5
                
                card_played = [c for c in exp['state'].hands[pid] if c.id == exp['action_id']][0]
                for reward_class in self.rewards:
                    reward += reward_class.calc_reward(exp['state'], pid, card_played) 
                
                if exp['done']:
                    for reward_class in self.final_rewards:
                        reward += reward_class.calc_reward(exp['state'], pid)
                    reward += player_contributions[pid] * 0.2
 
                    
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
                self.agent.epsilon = max(0.05, self.agent.epsilon * 0.95)
            
            if e % 100 == 0:
                avg_team0 = sum(s[0] for s in recent_scores) / len(recent_scores) if recent_scores else 0
                avg_team1 = sum(s[1] for s in recent_scores) / len(recent_scores) if recent_scores else 0
                
                print(f"Episode {e}/{episodes} | "
                    f"Score: T0={final_scores[0]} T1={final_scores[1]} | "
                    f"Avg(100): T0={avg_team0:.1f} T1={avg_team1:.1f} | "
                    f"Epsilon: {self.agent.epsilon:.3f}")
            
            if (e + 1) % 1000 == 0:
                RLAgentPersist.save(self, save_path, e+1)
                print(f"Model saved at episode {e+1}")

        print("\n=== Training Complete ===")
        RLAgentPersist.save(self, save_path, episodes)
