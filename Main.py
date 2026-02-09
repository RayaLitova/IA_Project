from BelotDQN import BelotDQN
from StateEncoder import StateEncoder
from GameState import GameState
from RLAgent import RLAgent
from Card import SUITS, RANKS, CONTRACTS, Card
import random
import copy
from BelotRules import BelotRules
import time

import torch
import os

def save(self, filepath='belot_model.pth'):
    checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'epsilon': self.epsilon,
        'input_size': self.encoder.input_size
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load(filepath='belot_model.pth'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file '{filepath}' not found!")
    
    checkpoint = torch.load(filepath)
    
    encoder = StateEncoder()
    model = BelotDQN(checkpoint['input_size'], 32)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    agent = RLAgent(model, encoder, epsilon=checkpoint['epsilon'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath} (epsilon={checkpoint['epsilon']:.3f})")
    return agent

def train_agent(episodes=100, save_path='belot_model.pth'):
    print(f"Training Neural Network for {episodes} games...")
    print("Team Setup: Player 0 (You) & Player 2 (Friendly AI) vs Player 1 & Player 3 (Opponents)")
    
    encoder = StateEncoder()
    model = BelotDQN(encoder.input_size, 32)
    agent = RLAgent(model, encoder, epsilon=0.5)
    print("Starting fresh training")
    
    deck = [Card(r, s) for s in SUITS for r in RANKS]

    for e in range(episodes):
        random.shuffle(deck)
        hands = {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(4)}
        contract = random.choice(CONTRACTS)
        
        state = GameState(contract, hands)
        
        game_experiences = []
        
        while not state.is_terminal():
            pid = state.get_current_player()
            
            curr_state_obj = copy.deepcopy(state)
            
            card = agent.get_action(state, pid, training=True)
            
            next_state, trick_rewards = state.apply_move(card)
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
        
        if final_scores[0] > final_scores[1]:
            game_bonus = [50, -50]
        elif final_scores[1] > final_scores[0]:
            game_bonus = [-50, 50]
        else:
            game_bonus = [0, 0]
        
        for exp in game_experiences:
            pid = exp['player_idx']
            team = pid % 2
            
            reward = exp['trick_rewards'][team]
            
            # penalize playing high cards when your team is not winning
            if len(exp['state'].current_trick) >= 1:
                partner_idx = (pid + 2) % 4
                current_winner_idx, _ = BelotRules.get_trick_winner(
                    exp['state'].trick_starter,
                    exp['state'].current_trick, 
                    exp['state'].contract
                )
                
                if current_winner_idx != partner_idx and current_winner_idx != pid:
                    card_played = [c for c in exp['state'].hands[pid] if c.id == exp['action_id']][0]
                    card_value = BelotRules.get_points(card_played, exp['state'].contract)
                    
                    if card_value >= 10:
                        reward -= 10
                        
            if exp['done']:
                reward += game_bonus[team]
            
            agent.remember(
                exp['state'],
                exp['action_id'],
                reward,
                exp['next_state'],
                exp['done'],
                pid
            )
        
        for _ in range(4):
            agent.replay()
            
        if e % 100 == 0:
            agent.epsilon = max(0.05, agent.epsilon * 0.85)
            print(f"Episode {e}/{episodes} | Final Score: Team0={final_scores[0]} Team1={final_scores[1]} | Epsilon: {agent.epsilon:.3f}")
        
        if (e + 1) % 1000 == 0:
            save(agent, save_path)

    print("Training Complete!")
    save(agent, save_path)
    return agent

def play_vs_ai(ai_agent):    
    deck = [Card(r, s) for s in SUITS for r in RANKS]
    random.shuffle(deck)
    hands = {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(4)}
    print("Your hand (Player 0):", hands[0])
    print("Available contracts: 0:AT, 1:NT, 2:♠, 3:♦, 4:♥, 5:♣")
    try:
        c_idx = int(input("Choose contract: "))
        contract = CONTRACTS[c_idx]
    except:
        contract = 'AT'
    
    state = GameState(contract, hands)
    
    print(f"\nContract: {contract}")
    print("You (Player 0) + Friendly AI (Player 2)")
    print("Opponent AI (Player 1) + Opponent AI (Player 3)")
    
    while not state.is_terminal():
        pid = state.get_current_player()
        
        if pid == 0: # player
            legal = BelotRules.get_valid_moves(pid, state.hands[0], state.trick_starter, state.current_trick, state.contract)
            print(f"\nTable: {state.current_trick if state.current_trick else 'Empty'}")
            print(f"Your Hand: {[f'{i}:{c}' for i,c in enumerate(legal)]}")
            idx = int(input("Choose card index: "))
            card = legal[idx]
        else:
            card = ai_agent.get_action(state, pid, training=False)
            player_name = "Partner" if pid == 2 else f"Opponent {pid}"
            print(f"Player {pid} ({player_name}) plays: {card}")
            time.sleep(0.3)
            
        state, _ = state.apply_move(card)
        
        if len(state.current_trick) == 0 and not state.is_terminal():
            print(f"Trick complete! Scores: Your Team {state.scores[0]} - {state.scores[1]} Opponent Team")
    
    print(f"\nGAME OVER! Final Score: Your Team {state.scores[0]} - {state.scores[1]} Opponent Team")
    if state.scores[0] > state.scores[1]:
        print("You WIN!")
    elif state.scores[1] > state.scores[0]:
        print("You LOSE!")
    else:
        print("It's a DRAW!")

if __name__ == "__main__":
    print("\n1. Train new model")
    print("2. Load model and play")
    
    choice = input("\nYour choice (1/2): ").strip()
    
    if choice == "1":
        agent = train_agent(episodes=5000, save_path='belot_model.pth')
        play_vs_ai(agent)

    elif choice == "2":
        try:
            agent = load('belot_model.pth')
            play_vs_ai(agent)
        except FileNotFoundError as e:
            print(e)
            print("Train a model first!")
    else:
        print("Invalid choice. Run the script again.")