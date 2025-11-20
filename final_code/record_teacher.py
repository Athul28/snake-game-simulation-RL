import pickle
from agent import Agent
from game import SnakeGameAI

def record_expert_data():
    expert_data = []
    agent = Agent()
    game = SnakeGameAI()

    print("Generating expert demonstration data...")

    # collect 5,000 state-action samples
    while len(expert_data) < 5000:
        state = agent.get_state(game)
        action = agent.get_action(state)

        expert_data.append((state, action))

        reward, done, score = game.play_step(action)

        if done:
            game.reset()

    with open("expert_data.pkl", "wb") as f:
        pickle.dump(expert_data, f)

    print("Saved expert_data.pkl with", len(expert_data), "samples")

record_expert_data()