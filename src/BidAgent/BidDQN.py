from BaseClasses.DQN import DQN

class BidDQN(DQN):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, 128, output_size)