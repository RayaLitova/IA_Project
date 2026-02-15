from Game.BelotPhase import BelotPhase
from Game.BidPhase import BidPhase
from Belot.BelotRules import BelotRules

class Game:
    def start(self):
        print("\n1. Train new model")
        print("2. Load model and play without bid")
        print("3. Train bid model")
        print("4. Load model and play with bid")
        
        choice = input("\nYour choice (1/2/3/4): ").strip()
        
        if choice == "1":
            self.play(True)
        elif choice == "2":
            try:
                self.play()
            except FileNotFoundError as e:
                print(e)
                print("Train a model first!")
        elif choice == "3":
            try:
                self.play_with_bid(True)
            except FileNotFoundError as e:
                print(e)
                print("Train a belot model first!")
        elif choice == "4":
            try:
                self.play_with_bid()
            except FileNotFoundError as e:
                print(e)
                print("Train a model first!")
        else:
            print("Invalid choice. Run the script again.")
            
    def play(self, train = False):
        rules = BelotRules()
        BelotPhase(rules).play(train)
    
    def play_with_bid(self, train = False):
        rules = BelotRules()
        hands, contract = BidPhase(rules).play(train)
        if not hands:
            print("Everyone passed. Start a new game")
            return
        BelotPhase(rules, hands, contract).play()
            
        