from flask import Flask, request, jsonify
from GameAgent.BelotState import GameState
from Belot.Card import RANKS, SUITS, Card
from Belot.BelotRules import BelotRules
from GameAgent.BelotRLAgent import BelotRLAgent
from BaseClasses.RLAgentPersist import RLAgentPersist
from GameAgent.BelotDQN import BelotDQN
from GameAgent.BelotStateEncoder import BelotStateEncoder

app = Flask(__name__)

def process_cards(cards : list[str]) -> list[Card]:
    cards_res = []
    for i in range(0, len(cards['suit'])):
        cards_res.append(Card(RANKS[cards['rank'][i]], SUITS[cards['suit'][i]]))
    return cards_res

@app.route('/process', methods=['POST'])
def process_data():
    data = request.json
    if data['contract'] == 2:
        contract = SUITS[data['trump_suit']]
    else:
        contract = BelotRules.CONTRACTS[data['contract']]
    state = GameState(BelotRules(), contract, 
                    [process_cards(data['hand0']), process_cards(data['hand1']), process_cards(data['hand2']), process_cards(data['hand3'])],
                    process_cards(data['played_cards']), 
                    process_cards(data['current_trick']), 
                    data['trick_starter_idx'], 
                    [data['team_scores']['team1'], data['team_scores']['team2']]
    )
    agent = BelotRLAgent(BelotDQN(106, 32), BelotStateEncoder())
    RLAgentPersist.load(agent, 'models/game/belot_model.pth')
    card = agent.get_action(state, data['player_idx'])
    return jsonify({'rank': RANKS.index(card.rank), 'suit': SUITS.index(card.suit)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)