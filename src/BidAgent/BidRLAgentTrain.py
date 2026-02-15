import copy
import random
import numpy as np
import torch
from BaseClasses.RLAgentTrain import RLAgentTrain
from BaseClasses.RLAgentPersist import RLAgentPersist
from BaseClasses.State import State
from Belot.Card import Card
from Belot.BelotRules import BelotRules
from GameAgent.BelotState import GameState
from BaseClasses.RLTrainReward import RLTrainReward, RLTrainRewardFinal
from BaseClasses.RLAgent import RLAgent

class BidRLAgentTrain(RLAgentTrain): 
    def __init__(self, bid_agent : RLAgent, belot_agent : RLAgent, rewards : list[RLTrainReward], final_rewards : list[RLTrainRewardFinal]):
        super().__init__(bid_agent, rewards, final_rewards)
        self.belot_agent = belot_agent
        
    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = []
        targets = []
        
        for state, action_id, reward, pid in batch:
            state_tensor = self.agent.encoder.encode(state, pid)
            target = reward
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
        print(f"Training Bid Neural Network for {episodes} games...")
        print("Starting fresh training with curriculum learning")
        
        biddable_contracts = [c for c in BelotRules.CONTRACTS if c != "Pass"]
        total_rewards = []
        
        for episode in range(episodes):
            hands = Card.deal_deck()
            player = random.randrange(0, 4)
            
            bid_state = State(hands, player, ["Pass"] * 4)
            bid_curr_state_obj = copy.deepcopy(bid_state)
            
            contract = self.agent.get_action(bid_state, player, training=True)
            
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
                        print(f"Episode {episode}/{episodes} | Agent passed (penalized) | Epsilon: {self.agent.epsilon:.3f}")
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
                    
                reward = 0
                for reward_class in self.final_rewards:
                    reward += reward_class.calc_reward(game_state, player)
                
                if forced_bid:
                    reward -= 20
                
                action_id = 6 if contract == "Pass" else BelotRules.CONTRACTS.index(contract)
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
                self.agent.epsilon = max(0.05, self.agent.epsilon * 0.995)
            
            if episode % 100 == 0:
                avg_reward = sum(total_rewards[-100:]) / min(len(total_rewards), 100)
                print(f"Episode {episode}/{episodes} | "
                    f"Avg Reward: {avg_reward:.1f} | "
                    f"Epsilon: {self.agent.epsilon:.3f} | "
                    f"Memory: {len(self.memory)}")
            
            if (episode + 1) % 1000 == 0:
                checkpoint_path = f"{save_path[:-4]}_ep{episode+1}.pth"
                RLAgentPersist.save(self, checkpoint_path, episode+1)
        
        avg_final_reward = sum(total_rewards[-1000:]) / min(len(total_rewards), 1000)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Final Avg Reward (last 1000): {avg_final_reward:.1f}")
        print(f"Final Epsilon: {self.agent.epsilon:.3f}")
        print("="*60)
        
        RLAgentPersist.save(self, save_path, episodes)