import torch.nn as nn
import torch
from BelotRules import BelotRules
import random
import numpy as np
from Card import Card, SUITS, RANKS, CONTRACTS
from torch import FloatTensor
import copy
from GameState import GameState
import torch.optim as optim
from collections import deque


class BelotBidDQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder produces 32 card features + 12 bid features = 44
        self.fc1 = nn.Linear(44, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 7)   # 7 possible bids
    
    def forward(self, hand_encoding):
        x = torch.relu(self.fc1(hand_encoding))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class BelotBidAgent:
    def __init__(self, model, encoder, epsilon=0.1):
        self.model = model
        self.encoder = encoder
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
    
    def get_bid(self, state, player_idx, training=False):
        legal_bids = BelotRules.get_legal_bids(state.current_bids)
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

    def remember(self, state, action_id, reward, player_idx):
        self.memory.append((state, action_id, reward, player_idx))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = []
        targets = []
        
        # Memory format: (state, action_id, reward, player_idx)
        for state, action_id, reward, pid in batch:
            state_tensor = self.encoder.encode(state, pid)
            
            # Since we don't have next_state, just use immediate reward
            target = reward
            
            # Get current Q-values
            current_qs = self.model(state_tensor).detach().clone()
            
            # Update only the Q-value for the action taken
            current_qs[action_id] = target
            
            states.append(state_tensor.numpy())
            targets.append(current_qs.numpy())

        # Convert to tensors
        input_batch = torch.FloatTensor(np.array(states))
        target_batch = torch.FloatTensor(np.array(targets))
        
        # Train the model
        self.optimizer.zero_grad()
        outputs = self.model(input_batch)
        loss = self.criterion(outputs, target_batch)
        loss.backward()
        self.optimizer.step()
        
class BidState:
    def __init__(self, hands, bid_starter, current_bids = None):
        self.hands = hands 
        self.current_bids = current_bids if current_bids else [] 
        self.bid_starter = bid_starter
        
    def get_current_player(self):
        return (self.bid_starter + len(self.current_bids)) % 4
        
def train_bid_agent(belot_agent, episodes=50000, save_path='bid_model.pth'):
    print(f"Training Bid Neural Network for {episodes} games...")
    
    # Initialize encoder, model, and agent
    encoder = BidStateEncoder()
    model = BelotBidDQN()
    agent = BelotBidAgent(model, encoder, epsilon=0.9)  # High exploration at start
    
    print("Starting fresh training with curriculum learning")
    
    # Create deck
    deck = [Card(r, s) for s in SUITS for r in RANKS]
    
    # Contracts without "Pass" for forced bidding phase
    biddable_contracts = [c for c in CONTRACTS if c != "Pass"]
    
    # Training statistics
    total_rewards = []
    win_count = 0
    
    for episode in range(episodes):
        # Shuffle and deal cards
        random.shuffle(deck)
        hands = {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(4)}
        
        # Choose a random player to bid
        player = random.randrange(0, 4)
        
        # Create bid state (no previous bids)
        bid_state = BidState(hands, player, ["Pass"] * 4)
        bid_curr_state_obj = copy.deepcopy(bid_state)
        
        # Agent makes a bid decision
        contract = agent.get_bid(bid_state, player, training=True)
        
        # Curriculum Learning - 3 phases
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
                agent.remember(bid_curr_state_obj, 0, penalty, player)
                
                if episode % 100 == 0:
                    print(f"Episode {episode}/{episodes} | Agent passed (penalized) | Epsilon: {agent.epsilon:.3f}")
                continue
        
        # Phase 3 (7000+): Normal training with moderate pass penalty
        else:
            if not contract or contract == "Pass":
                penalty = -30
                agent.remember(bid_curr_state_obj, 0, penalty, player)
                continue
        
        # Play the full game with the chosen contract
        try:
            game_state = GameState(contract, hands)
            
            while not game_state.is_terminal():
                current_player = game_state.get_current_player()
                card = belot_agent.get_action(game_state, current_player)
                next_state_tuple = game_state.apply_move(card)
                game_state, _ = next_state_tuple
            
            # Game finished - calculate reward
            final_scores = game_state.scores
            
            # Determine which team the bidding player is on (0 or 1)
            team = player % 2
            opponent_team = 1 - team
            
            # Calculate point difference
            team_score = final_scores[team]
            opponent_score = final_scores[opponent_team]
            point_diff = team_score - opponent_score
            
            # Reward structure
            if point_diff > 0:
                # Won the game
                base_reward = 100
                # Bonus for winning by large margin
                margin_bonus = min(point_diff / 5, 50)  # Cap at 50
                reward = base_reward + margin_bonus
                win_count += 1
            elif point_diff < 0:
                # Lost the game
                base_reward = -100
                # Extra penalty for losing badly
                margin_penalty = max(point_diff / 5, -50)  # Cap at -50
                reward = base_reward + margin_penalty
            else:
                # Draw
                reward = 0
            
            # Penalty for being forced to bid
            if forced_bid:
                reward -= 20
            
            # Store experience in memory
            action_id = 6 if contract == "Pass" else CONTRACTS.index(contract)
            agent.remember(bid_curr_state_obj, action_id, reward, player)
            
            # Track statistics
            total_rewards.append(reward)
            
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            # Give negative reward for causing errors
            agent.remember(bid_curr_state_obj, 0, -50, player)
            continue
        
        # Train the agent multiple times per episode
        training_iterations = 8 if episode < 5000 else 4
        for _ in range(training_iterations):
            agent.replay()
        
        # Epsilon decay - slower decay for better exploration
        if episode % 50 == 0:
            agent.epsilon = max(0.05, agent.epsilon * 0.995)
        
        # Progress logging
        if episode % 100 == 0:
            avg_reward = sum(total_rewards[-100:]) / min(len(total_rewards), 100)
            win_rate = win_count / (episode + 1) * 100
            print(f"Episode {episode}/{episodes} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Win Rate: {win_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Memory: {len(agent.memory)}")
        
        # Save checkpoint
        if (episode + 1) % 1000 == 0:
            checkpoint_path = f"{save_path[:-4]}_ep{episode+1}.pth"
            save(agent, checkpoint_path, episode=episode+1)
    
    # Final statistics
    final_win_rate = win_count / episodes * 100
    avg_final_reward = sum(total_rewards[-1000:]) / min(len(total_rewards), 1000)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final Win Rate: {final_win_rate:.1f}%")
    print(f"Final Avg Reward (last 1000): {avg_final_reward:.1f}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print("="*60)
    
    # Save final model
    save(agent, save_path, episode=episodes)
    
    return agent

class BidStateEncoder:
    def __init__(self):
        # My Hand (32) + CurrentBids (12)
        self.input_size = 32 + 12

    def encode(self, state, player_idx):
        hand_vec = np.zeros(32)
        for card in state.hands[player_idx]:
            hand_vec[card.id] = 1
        # CurrentBids encoding (12):
        # - first 6 entries: presence (has this contract been bid at least once)
        # - next 6 entries: one-hot of the last non-Pass contract (most recent bidder)
        current_bids_vec = np.zeros(12)
        # presence
        for b in state.current_bids:
            if b != "Pass" and b in CONTRACTS:
                idx = CONTRACTS.index(b)
                current_bids_vec[idx] = 1

        # last non-pass one-hot
        last_non_pass = None
        for b in reversed(state.current_bids):
            if b != "Pass":
                last_non_pass = b
                break
        if last_non_pass and last_non_pass in CONTRACTS:
            current_bids_vec[6 + CONTRACTS.index(last_non_pass)] = 1

        full_vec = np.concatenate([hand_vec, current_bids_vec])
        return FloatTensor(full_vec)
    
def save(agent, filepath='bid_model.pth', episode=None):
    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'episode': episode,
        'memory_size': len(agent.memory)
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")
    

