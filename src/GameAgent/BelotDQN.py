from BaseClasses.DQN import DQN

class BelotDQN(DQN):
    def __init__(self, input_size : int, output_size : int):
        super().__init__(input_size, 256, output_size)