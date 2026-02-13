from BaseClasses.DQN import DQN

class BidDQN(DQN):
    def __init__(self, input_size : int, output_size : int):
        super().__init__(input_size, 128, output_size)