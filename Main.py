from BelotDQN import BelotDQN
from StateEncoder import StateEncoder
from GameState import GameState
from RLAgent import RLAgent
from Card import SUITS, RANKS, CONTRACTS, Card, ORDER
import random
import copy
from BelotRules import BelotRules
import time
import BidAgent

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

def load_bid_model(filepath='bid_model.pth'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file '{filepath}' not found!")
    
    checkpoint = torch.load(filepath)
    encoder = BidAgent.BidStateEncoder()
    model = BidAgent.BelotBidDQN()
    model.load_state_dict(checkpoint['model_state_dict'])
    agent = BidAgent.BelotBidAgent(model, encoder, epsilon=checkpoint.get('epsilon', 0.1))
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath} (epsilon={checkpoint.get('epsilon', 0.1):.3f})")
    return agent

def train_agent(episodes=50000, save_path='belot_model.pth', load_existing=False):
    print(f"Training Neural Network for {episodes} games...")
    print("Team Setup: Player 0 (You) & Player 2 (Friendly AI) vs Player 1 & Player 3 (Opponents)")
    
    encoder = StateEncoder()
    model = BelotDQN(encoder.input_size, 32)
    
    if load_existing and os.path.exists(save_path):
        agent = load(save_path)
        print(f"Loaded existing model from {save_path}")
    else:
        agent = RLAgent(model, encoder, epsilon=0.9)
        print("Starting fresh training")
    
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
            
            card = agent.get_action(state, pid, training=True)
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
            
            
            if len(exp['state'].current_trick) >= 1:
                current_winner_idx, _ = BelotRules.get_trick_winner(
                    exp['state'].trick_starter,
                    exp['state'].current_trick, 
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
                    
                    temp_trick = exp['state'].current_trick + [card_played]
                    new_winner_idx, _ = BelotRules.get_trick_winner(
                        exp['state'].trick_starter,
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
                
                temp_trick = exp['state'].current_trick + [card_played]
                winner_idx, _ = BelotRules.get_trick_winner(
                    exp['state'].trick_starter,
                    temp_trick,
                    exp['state'].contract
                )
                
                if winner_idx != pid and winner_idx != partner_idx and not has_protection:
                    reward -= 40
                elif len(exp['state'].current_trick) == 0 and not has_protection:
                    reward -= 30
                elif has_protection:
                    reward += 8
            
            lead_suit = exp['state'].current_trick[0].suit if exp['state'].current_trick else None
            if lead_suit and card_played.suit != lead_suit:
                has_lead_suit = any(c.suit == lead_suit for c in exp['state'].hands[pid])
                if has_lead_suit:
                    reward -= -float('inf')
            
            if card_played.suit == exp['state'].contract:
                if len(exp['state'].current_trick) >= 1:
                    current_winner_idx, _ = BelotRules.get_trick_winner(
                        exp['state'].trick_starter,
                        exp['state'].current_trick,
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
            
            agent.remember(
                exp['state'],
                exp['action_id'],
                reward,
                exp['next_state'],
                exp['done'],
                pid
            )
        
        replay_times = 4 if e < 1000 else 8 if e < 5000 else 16
        for _ in range(replay_times):
            agent.replay()
        
        if e % 50 == 0:
            agent.epsilon = max(0.05, agent.epsilon * 0.95)
        
        if e % 100 == 0:
            avg_team0 = sum(s[0] for s in recent_scores) / len(recent_scores) if recent_scores else 0
            avg_team1 = sum(s[1] for s in recent_scores) / len(recent_scores) if recent_scores else 0
            win_rate = team0_wins / (e + 1) * 100
            
            print(f"Episode {e}/{episodes} | "
                  f"Score: T0={final_scores[0]} T1={final_scores[1]} | "
                  f"Avg(100): T0={avg_team0:.1f} T1={avg_team1:.1f} | "
                  f"Win Rate: {win_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        if (e + 1) % 1000 == 0:
            save(agent, save_path)
            print(f"Model saved at episode {e+1}")

    print("\n=== Training Complete ===")
    print(f"Team 0 Wins: {team0_wins} ({team0_wins/episodes*100:.1f}%)")
    print(f"Team 1 Wins: {team1_wins} ({team1_wins/episodes*100:.1f}%)")
    save(agent, save_path)
    return agent

def play_vs_ai_with_bid(bid_agent, belot_agent):
    deck = [Card(r, s) for s in SUITS for r in RANKS]
    random.shuffle(deck)
    hands = {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(4)}
    bid_state = BidAgent.BidState(hands, 0, [])
    bid_player_idx = 0
    while BelotRules.get_legal_bids(bid_state.current_bids):
        if bid_player_idx == 0:
            print("Your hand (Player 0):", hands[0])
            print("Available contracts: 0:AT, 1:NT, 2:♠, 3:♦, 4:♥, 5:♣, 6:Pass")
            try:
                c_idx = int(input("Choose contract: "))
                contract = "Pass" if c_idx == 6 else CONTRACTS[c_idx]
            except:
                contract = 'AT'
            bid_state.current_bids += [contract]
        else:
            bid_state.current_bids += [bid_agent.get_bid(bid_state, bid_player_idx)]
            print(f"Player {bid_player_idx} bid {bid_state.current_bids[-1]}")
        bid_player_idx = (bid_player_idx + 1) % 4
            
    contract = [b for b in bid_state.current_bids if b != "Pass"]
    if not contract:
        return
    print("Final contract:" + contract[-1])
    play_vs_ai(belot_agent, hands, contract[-1])
    

def play_vs_ai(ai_agent, hands = None, contract = None): 
    if not hands:   
        deck = [Card(r, s) for s in SUITS for r in RANKS]
        random.shuffle(deck)
        hands = {i: sorted(deck[i*8:(i+1)*8], key=lambda c: c.id) for i in range(4)}
    
    if not contract:
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
    print("2. Load model and play without bid")
    print("3. Train bid model")
    print("4. Load model and play with bid")
    
    choice = input("\nYour choice (1/2/3/4): ").strip()
    
    if choice == "1":
        agent = train_agent()
        play_vs_ai(agent)

    elif choice == "2":
        try:
            agent = load('belot_model.pth')
            play_vs_ai(agent)
        except FileNotFoundError as e:
            print(e)
            print("Train a model first!")
    elif choice == "3":
        try:
            agent = load('belot_model.pth')
            bid_agent = BidAgent.train_bid_agent(agent)
            play_vs_ai_with_bid(bid_agent, agent)
        except FileNotFoundError as e:
            print(e)
            print("Train a belot model first!")
    elif choice == "4":
        try:
            belot_agent = load('belot_model.pth')
            bid_agent = load_bid_model('bid_model.pth')
            play_vs_ai_with_bid(bid_agent, belot_agent)
        except FileNotFoundError as e:
            print(e)
            print("Train a model first!")
    else:
        print("Invalid choice. Run the script again.")