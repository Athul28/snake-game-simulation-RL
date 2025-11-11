# Snake RL (Final Code)

This folder contains the final, self-contained implementation used to train a Deep Q-Learning agent to play the classic Snake game.

The implementation is intentionally small and educational: it demonstrates how to wire a simple environment (built with Pygame) to a tiny PyTorch neural network and a basic DQN training loop.

## Contents

- `agent.py` — training loop and the RL Agent class. Handles experience replay, action selection (epsilon-greedy), short- and long-term training steps, and coordinates episodes.
- `game.py` — Pygame-based Snake environment (`SnakeGameAI`). Exposes `play_step(action)` which returns `(reward, done, score)`. Also contains `Direction` and `Point` utility types.
- `helper.py` — small utility for plotting training progress. Uses a headless Matplotlib backend and saves a PNG file (`training_curve.png`) during training.
- `model.py` — defines the PyTorch model (`Linear_QNet`) and the `QTrainer` helper with the `train_step` method (optimizer, loss calculation, target Q updates).
- `model/` — directory where trained model weights are saved (default file: `model/model.pth`).

## Quick overview

This project uses a reinforcement learning approach (Deep Q-Learning) with these main pieces:

- Environment (`SnakeGameAI`): returns observations (the agent's `get_state` function in `agent.py` constructs an 11-dim binary state), reward, and done flag.
- Policy model (`Linear_QNet`): a small 2-layer feed-forward network (11 -> 256 -> 3) predicting Q-values for the three discrete actions: [straight, right, left].
- Replay memory: a deque capped at `MAX_MEMORY` (see `agent.py`). Training uses mini-batches drawn from this memory.
- Trainer (`QTrainer`): computes targets and performs gradient descent using Adam and MSE loss.

## File-level details

### `agent.py`

- Constants at top:

  - `MAX_MEMORY = 100_000` — maximum number of experiences kept in replay buffer.
  - `BATCH_SIZE = 1000` — samples per training step (if reached).
  - `LR = 0.001` — learning rate for the Adam optimizer in `QTrainer`.

- `Agent` class:

  - `get_state(game)` — constructs an 11-element binary state vector describing: immediate dangers (straight/right/left), current direction (left/right/up/down), and food relative position (left/right/up/down).
  - `remember(...)` — append experience tuples to replay buffer.
  - `train_long_memory()` — samples a minibatch from memory and calls `trainer.train_step` on the batch.
  - `train_short_memory(...)` — trains on the most recent transition (useful for faster learning from the latest move).
  - `get_action(state)` — epsilon-greedy action selection. Epsilon decreases with game count to favor exploitation over time.

- `train()` function: main loop that instantiates `Agent` and `SnakeGameAI`, runs episodes, records scores, saves the model whenever a new high score is reached, and calls `helper.plot` to save the training curve.

Run training with:

```bash
# from repository root
python3 final_code/agent.py
```

This starts training and opens the Pygame window (the game runs visually). The training plot will be saved as `training_curve.png` in the working directory and the model checkpoint will be saved under `model/model.pth` when a new record is achieved.

### `game.py`

- `SnakeGameAI` implements the game logic using Pygame. Important behaviors:
  - `play_step(action)` applies an action (one-hot list `[straight, right, left]`), moves the snake, detects collisions, updates score and places food.
  - `is_collision(pt=None)` detects wall collisions and self-collisions.
  - `_place_food()` randomly places food not overlapping the snake.
  - Fast mode: press the `F` key while the game window has focus to toggle fast mode. This increases the Pygame clock tick from `SPEED_NORMAL` to `SPEED_FAST` so training episodes proceed much quicker visually.

Note: the coordinate/grid system uses `BLOCK_SIZE = 20`, and the window defaults to 640×480.

### `helper.py`

- Provides `plot(scores, mean_scores, save_path='training_curve.png')` which saves a PNG showing per-episode scores and running mean. It uses Matplotlib's `Agg` backend so training can run headless and still persist a plot file.

### `model.py`

- `Linear_QNet` — a small 2-layer network with ReLU activation.

  - `save(file_name='model.pth')` writes model weights to `./model/model.pth` (creates `model/` if necessary).

- `QTrainer` — encapsulates the training step (creates tensors, computes predicted Q-values from the model, sets up the target values using reward + gamma \* max(next_state Q), computes MSE loss, and performs a single optimizer step).

## Inputs / Outputs (contract)

- Inputs:

  - Observations: 11-element binary numpy array from `Agent.get_state(game)`.
  - Actions: one-hot lists of length 3: [straight, right, left].

- Outputs:
  - Trained model weights saved to `model/model.pth`.
  - Training plot saved to `training_curve.png` (by default in the working directory).

## Installation / Requirements

Install dependencies listed in the repository root `requrements.txt` (note: file name is `requrements.txt` in this repo). Example using pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requrements.txt
```

Key packages: `torch`, `pygame`, `matplotlib`, `numpy`

## How to run

- Train the agent (visual game window + training):

```bash
python3 final_code/agent.py
```

- After (or during) training you should see `training_curve.png` and the model saved to `model/model.pth` when a new record is reached.

Notes on running:

- The Pygame window must have focus to receive the `F` key toggle for fast mode. Fast mode is useful to speed up visual training.
- If you want fully headless training without opening a visible window, you would need to adapt `game.py` to avoid creating a Pygame display (currently `SnakeGameAI` constructs a display). Alternatively run on a machine with a virtual frame buffer (Xvfb) or modify the code to skip rendering during training runs.

