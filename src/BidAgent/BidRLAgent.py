import copy
import random
import numpy as np
import torch
from BaseClasses.RLAgent import RLAgent
from BaseClasses.State import State 
from Belot.BelotRules import BelotRules
from Belot.Card import CONTRACTS, RANKS, SUITS, Card
from BidAgent.BidDQN import BidDQN
from BidAgent.BidStateEncoder import BidStateEncoder
from GameAgent.BelotState import GameState

class BidRLAgent(RLAgent):
    def __init__(self, belot_agent):
        super().__init__(BidDQN(44, 7), BidStateEncoder(), epsilon = 0.9)
        self.belot_agent = belot_agent
        
    def get_action(self, state, player_idx, training=False):
        legal_bids = BelotRules.get_legal_bids(state.played_moves)
        if not legal_bids: 
            return
        if training and random.random() < self.epsilon:
            return random.choice(legal_bids)

        state_tensor = self.encoder.encode(state, player_idx)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        legal_ids = [6 if c == "Pass" else CONTRACTS.index(c) for c in legal_bids]
        mask = torch.full_like(q_values, -float('inf'))
        mask[legal_ids] = 0
        
        masked_q_values = q_values + mask
        best_bid_id = torch.argmax(masked_q_values).item()
        
        for b in legal_ids:
            if b == best_bid_id:
                return "Pass" if b == 6 else CONTRACTS[b]
        return legal_bids[0]

    #remember: state, action_id, reward, player_idx

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = []
        targets = []
        
        for state, action_id, reward, pid in batch:
            state_tensor = self.encoder.encode(state, pid)
            target = reward
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
    
    def train(self, episodes, save_path):
        print(f"Training Bid Neural Network for {episodes} games...")
        print("Starting fresh training with curriculum learning")
        
        deck = [Card(r, s) for s in SUITS for r in RANKS]
        biddable_contracts = [c for c in CONTRACTS if c != "Pass"]
        
        total_rewards = []
        win_count = 0
        
        for episode in range(episodes):
            random.shuffle(deck)
            hands = {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(4)}
            
            player = random.randrange(0, 4)
            
            bid_state = State(hands, player, ["Pass"] * 4)
            bid_curr_state_obj = copy.deepcopy(bid_state)
            
            contract = self.get_action(bid_state, player, training=True)
            
            forced_bid = False
            
            # Phase 1 (0-4000): Force agent to bid (no Pass allowed)
            if episode < 4000:
                if not contract or contract == "Pass":
                    contract = random.choice(biddable_contracts)
                    forced_bid = True
            
            # Phase 2 (4000-7000): Heavily penalize passing
            elif episode < 7000:
                if not contract or contract == "Pass":
                    penalty = -100
                    self.remember((bid_curr_state_obj, 0, penalty, player))
                    
                    if episode % 100 == 0:
                        print(f"Episode {episode}/{episodes} | Agent passed (penalized) | Epsilon: {self.epsilon:.3f}")
                    continue
            
            # Phase 3 (7000+): Normal training with moderate pass penalty
            else:
                if not contract or contract == "Pass":
                    penalty = -30
                    self.remember((bid_curr_state_obj, 0, penalty, player))
                    continue
            
            # Play the full game with the chosen contract
            try:
                game_state = GameState(contract, hands)
                
                while not game_state.is_terminal():
                    current_player = game_state.get_current_player()
                    card = self.belot_agent.get_action(game_state, current_player)
                    next_state_tuple = game_state.apply_move(card)
                    game_state, _ = next_state_tuple
                
                final_scores = game_state.scores
                team = player % 2
                opponent_team = 1 - team
                
                team_score = final_scores[team]
                opponent_score = final_scores[opponent_team]
                point_diff = team_score - opponent_score
                
                if point_diff > 0:
                    base_reward = 100
                    margin_bonus = min(point_diff / 5, 50)
                    reward = base_reward + margin_bonus
                    win_count += 1
                elif point_diff < 0:
                    base_reward = -100
                    margin_penalty = max(point_diff / 5, -50)
                    reward = base_reward + margin_penalty
                else:
                    reward = 0
                
                if forced_bid:
                    reward -= 20
                
                action_id = 6 if contract == "Pass" else CONTRACTS.index(contract)
                self.remember((bid_curr_state_obj, action_id, reward, player))
                
                total_rewards.append(reward)
                
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                self.remember((bid_curr_state_obj, 0, -50, player))
                continue
            
            training_iterations = 8 if episode < 5000 else 4
            for _ in range(training_iterations):
                self.replay()
            
            if episode % 50 == 0:
                self.epsilon = max(0.05, self.epsilon * 0.995)
            
            if episode % 100 == 0:
                avg_reward = sum(total_rewards[-100:]) / min(len(total_rewards), 100)
                win_rate = win_count / (episode + 1) * 100
                print(f"Episode {episode}/{episodes} | "
                    f"Avg Reward: {avg_reward:.1f} | "
                    f"Win Rate: {win_rate:.1f}% | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"Memory: {len(self.memory)}")
            
            if (episode + 1) % 1000 == 0:
                checkpoint_path = f"{save_path[:-4]}_ep{episode+1}.pth"
                self.save(checkpoint_path, episode+1)
        
        final_win_rate = win_count / episodes * 100
        avg_final_reward = sum(total_rewards[-1000:]) / min(len(total_rewards), 1000)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Final Win Rate: {final_win_rate:.1f}%")
        print(f"Final Avg Reward (last 1000): {avg_final_reward:.1f}")
        print(f"Final Epsilon: {self.epsilon:.3f}")
        print("="*60)
        
        self.save(save_path, episodes)