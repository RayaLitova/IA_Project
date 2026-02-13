from BaseClasses.DQN import DQN

class BelotDQN(DQN):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, 256, output_size)