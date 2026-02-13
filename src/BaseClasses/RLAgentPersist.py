import os
import torch
from BaseClasses.RLAgentTrain import RLAgentTrain
from BaseClasses.RLAgent import RLAgent

class RLAgentPersist:
    @staticmethod
    def save(trainer : RLAgentTrain, filepath : str, episode : int) -> None:
        checkpoint = {
            'model_state_dict': trainer.agent.model.state_dict(),
            'optimizer_state_dict': trainer.agent.optimizer.state_dict(),
            'epsilon': trainer.agent.epsilon,
            'episode': episode,
            'memory_size': len(trainer.memory)
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(agent : RLAgent, filepath : str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found!")
        
        checkpoint = torch.load(filepath)
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {filepath} (epsilon={checkpoint.get('epsilon', 0.1):.3f})")