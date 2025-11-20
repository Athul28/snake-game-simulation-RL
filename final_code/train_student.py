# train_student.py
import os
import torch
from student_agent import StudentAgent
from game import SnakeGameAI
from helper import plot

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = StudentAgent()
    game = SnakeGameAI()

    # Optional: create model directory early (trainer/model.save will handle too)
    if not os.path.exists("./model"):
        os.makedirs("./model")

    while True:
        # 1) get current state
        state_old = agent.get_state(game)

        # 2) choose action a (for SARSA we need to pick next_action too)
        action_old = agent.get_action(state_old)

        # 3) perform action a -> observe r, s'
        reward, done, score = game.play_step(action_old)
        state_new = agent.get_state(game)

        # 4) choose next action a' using current policy (epsilon-greedy)
        action_new = agent.get_action(state_new)

        # 5) train short memory with SARSA update: (s, a, r, s', a')
        agent.train_short_memory(state_old, action_old, reward, state_new, action_new, done)

        # 6) store transition for experience replay
        agent.remember(state_old, action_old, reward, state_new, action_new, done)

        # 7) when episode ends, perform long-memory training and bookkeeping
        if done:
            game.reset()
            agent.n_games += 1

            # train on a batch of SARSA transitions
            agent.train_long_memory()

            # save model if a new record
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # plotting (save-only using helper.plot)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()