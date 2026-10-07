# Pac-Man Reinforcement Learning (DVA514)

This repository explores reinforcement learning (RL) for an autonomous agent playing Pac-Man. It contains a Pygame implementation of the game and several experimental approaches to learning a policy, represented by different project branches/snapshots:

- **Base Pac-Man:** the game environment and mechanics.
- **Approximate Q-Learning (AQL):** a feature-based agent with manually designed state/action features and learned feature weights.
- **Feature-vector Deep Q-Learning:** a neural-network agent that receives a compact vector describing Pac-Man, pellets, ghosts, and the current episode.
- **CNN-based Double DQN:** a convolutional network that learns from processed game frames rather than hand-crafted state features.

Not every implementation is present in every branch. The filenames and commands below refer to the branch/snapshot where that implementation exists. Check out the relevant branch before running its code.

## Contents

- [Project goals](#project-goals)
- [Repository and branch overview](#repository-and-branch-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the base game](#run-the-base-game)
- [Approximate Q-Learning](#approximate-q-learning)
- [Feature-vector Deep Q-Learning](#feature-vector-deep-q-learning)
- [CNN-based Double DQN](#cnn-based-double-dqn)
- [Evaluation and plots](#evaluation-and-plots)
- [Reported results](#reported-results)
- [Limitations and future work](#limitations-and-future-work)
- [Project structure](#project-structure)
- [Credits](#credits)

## Project goals

Pac-Man is a useful RL environment because an agent must collect pellets while navigating a maze, avoiding ghosts, and responding to changing game conditions. The project compares approaches with different state representations and model complexity.

The main research question in the accompanying report is which approach performs best when evaluated using metrics such as pellets collected and survival time. The report evaluates agents over 50 episodes and compares them with a random-action baseline.

## Repository and branch overview

The uploaded repository snapshots represent different stages of development. They should be understood as alternative implementations, not as folders that are guaranteed to coexist in one checkout.

| Implementation | Main files | State representation | Learning approach |
|---|---|---|---|
| Base game | `Pacman/run.py` | Game state used by the game engine | Manual keyboard control |
| Approximate Q-Learning | `Pacman/agentPac.py` (in the relevant branch) | Maze node, available directions, and hand-crafted features | Linear feature-weight updates |
| Feature-vector DQN | `Pacman/env.py`, `Pacman/DQN.py`, `Pacman/train_dqn.py`, `Pacman/run_dqn.py` (in the relevant branch) | Numeric vector containing position, direction, pellet, ghost, and timestep information | Fully connected Double DQN |
| CNN-based Double DQN | `Pacman_DQN/dqn_agent.py`, `Pacman_DQN/run.py` | Four stacked grayscale frames | Convolutional Double DQN |

The `Pacman` folder contains the game engine and assets, including maze layouts, sprites, Pac-Man, ghosts, pellets, fruit, and game-state logic. Depending on the branch, RL-specific files may be added to `Pacman`, or the CNN implementation may live in a separate `Pacman_DQN` directory.

## Requirements

The projects use Python 3. The required packages depend on the implementation you want to run:

- **Base game / AQL:** `pygame` (and Python standard-library modules).
- **Feature-vector DQN:** `pygame`, `numpy`, and `tensorflow`.
- **CNN-based DQN and evaluation scripts:** `pygame`, `numpy`, `tensorflow`, and `matplotlib`.

The supplied code snapshots were developed with different Python versions, so use a virtual environment and install compatible package versions if a dependency error occurs. TensorFlow compatibility can depend on your Python version and operating system.

## Installation

Clone the repository and check out the branch containing the implementation you want to use:

```bash
git clone https://github.com/OsirisPrime/DVA514.git
cd DVA514
# Optional: list available branches
git branch -a
# Switch to the branch you need
git switch <branch-name>
```

Create and activate a virtual environment (example for Linux/macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies for the selected implementation:

```bash
pip install pygame numpy tensorflow matplotlib
```

If you only want to run the base game, installing Pygame may be sufficient:

```bash
pip install pygame
```

## Run the base game

In a checkout containing the base game under `Pacman/`:

```bash
cd Pacman
python run.py
```

The game opens a Pygame window. In the manual-play version, use the arrow keys to move Pac-Man. The space bar pauses/resumes the game where supported by that version.

The environment includes maze navigation, pellets and power pellets, four ghosts, fruit, scoring, lives, and level/game-over handling. Game configuration such as tile dimensions, frame rate, and number of lives is defined in `constants.py`.

## Approximate Q-Learning

In the branch containing `Pacman/agentPac.py`, `AgentP` implements a feature-based Approximate Q-Learning agent.

### State and features

The discrete state includes Pac-Man's current maze-node coordinates and whether a neighboring node is available in each direction. The agent then calculates action features, including:

- Bias term
- Distance to the nearest pellet
- Distance to fruit, when present
- Ghost proximity / risk
- Ghost directly in the chosen direction
- Whether ghosts are frightened
- Action indicators for up, down, left, and right

The estimated action value is a weighted sum of features. Weights are updated using a temporal-difference error, the learning rate (`alpha`), and the discount factor (`gamma`). Exploration is controlled by `epsilon`.

The class defaults are `alpha=0.2`, `epsilon=0.2`, and `gamma=0.9`. Its reward method assigns penalties/rewards for death, eating a ghost, collecting fruit, collecting pellets, and otherwise taking a step. The implementation also provides a `save_weights()` method that writes learned feature weights to a CSV file.

**Running note:** the exact integration between `AgentP` and the game loop is branch-specific. Use the entry point and configuration from the same branch as `agentPac.py`; do not assume that the base-game `run.py` automatically starts training unless that branch's code explicitly connects the agent.

## Feature-vector Deep Q-Learning

In the branch containing `env.py`, `DQN.py`, `train_dqn.py`, and `run_dqn.py`, the game is wrapped by a lightweight `PacmanEnv` class.

### State representation

The environment builds a numeric `float32` vector containing:

- Normalized Pac-Man position
- Current direction encoded as one-hot values
- Vector and distance information for the nearest pellet
- Ghost information, including nearest-ghost and ghost-position features
- Normalized step count

The environment maps four discrete actions to up, down, left, and right. It advances the game between decision points, calculates a reward, and ends an episode when the game is over, the board is cleared, or the step limit is reached.

### Network and training

The Q-network is a fully connected neural network:

```text
Feature vector
     ↓
Dense(256, ReLU)
     ↓
Dense(256, ReLU)
     ↓
Dense(4) — Q-values for the four actions
```

The agent uses:

- Experience replay (default buffer capacity: 100,000 transitions)
- Online and target networks
- Double DQN action selection and target evaluation
- Epsilon-greedy exploration
- Adam optimization
- Huber loss

### Train

From the `Pacman` directory in the branch containing these files:

```bash
python train_dqn.py
```

The script defaults to 1,000 episodes. It saves weights to `dqn_pacman.weights.h5` and metadata such as epsilon and global step to `dqn_meta.npz`, and loads these files if they already exist so training can continue.

### Run a trained model

```bash
python run_dqn.py
```

This loads `dqn_pacman.weights.h5` and runs a greedy episode with epsilon set to zero. The weights file must exist before running the script.

### Test the environment wrapper

```bash
python test_env.py
```

This runs the environment with random actions and prints state dimensions, rewards, and episode information. It is a diagnostic script, rather than a trained-agent evaluation.

## CNN-based Double DQN

In the branch containing `Pacman_DQN/`, the agent learns from visual game input instead of the hand-crafted feature vector used in the earlier DQN implementation.

### Training

```bash
cd Pacman_DQN
python run.py
```

The main settings are in `constants.py`:

```python
USE_AI = True
MODEL_TRAINING = True
RENDER_MODE = True
```

- `USE_AI`: let the AI control Pac-Man; set to `False` for keyboard control.
- `MODEL_TRAINING`: changes game-loop behavior for training (including pauses and text overlays).
- `RENDER_MODE`: controls rendering. The checked-in constants note that headless rendering is not currently working reliably in this version, so leave it enabled unless you have verified otherwise.

During AI training, the `S` key saves a checkpoint immediately and `Q` saves and quits.

### Input and network architecture

The screen is converted to an 84 × 84 grayscale image. Four consecutive frames are stacked to form an `84 × 84 × 4` state. The CNN is structured as follows:

```text
Input: 84 × 84 × 4
  Conv2D: 32 filters, 8 × 8 kernel, stride 4, ReLU
  Conv2D: 64 filters, 4 × 4 kernel, stride 2, ReLU
  Conv2D: 64 filters, 3 × 3 kernel, stride 1, ReLU
  Flatten
  Dense: 512 units, ReLU
  Dense: 4 outputs (Q-values)
```

The four outputs correspond to up, down, left, and right. The implementation uses Double DQN, experience replay, a target network, epsilon-greedy exploration, Adam, Huber loss, and gradient clipping.

### Checkpoints and resuming

Checkpoints are stored in `Pacman_DQN/checkpoints/`. A checkpoint may include:

- `.weights.h5`: neural-network weights
- `.meta.json`: training metadata
- `.buffer.npz`: replay-buffer data, where saved for that checkpoint

To resume from a specific checkpoint, set `CHECKPOINT` in `Pacman_DQN/run.py` to the desired checkpoint path. Check which checkpoints have matching metadata and replay-buffer files before relying on a full training resume; not every saved checkpoint necessarily has all companion files.

### Greedy evaluation

Run the separate evaluation program:

```bash
python eval_run.py
```

It automatically searches for the latest available checkpoint by default. Options include:

```bash
python eval_run.py --checkpoint checkpoints/dqn_step_500000.weights.h5
python eval_run.py --episodes 50
python eval_run.py --no-render
```

The evaluation agent uses a greedy policy (`epsilon = 0`) and does not update the model during evaluation. Run these commands from the `Pacman_DQN` directory.

## Evaluation and plots

The CNN training-history plotting script reads checkpoint metadata and generates summary statistics and plots:

```bash
python eval.py
```

Useful options include:

```bash
python eval.py --smooth 100
python eval.py --checkpoint checkpoints/dqn_step_500000.weights.h5
python eval.py --all
python eval.py --out results
```

The output filename stem defaults to `results`, producing a PNG plot and PDF report. The repository also contains an example `results.png`, `results.pdf`, and `results_stats.txt` from a previous training run.

Keep in mind that **training-history plots and greedy evaluation results are different measurements**. Use the same evaluation protocol and episode count when comparing agents.

## Reported results

The accompanying report compares AQL, feature-based Deep Q-Learning, and CNN-based DQN over 50 evaluation episodes against a random baseline. The reported mean pellets collected were:

| Agent | Mean pellets | Percentage of 244 pellets |
|---|---:|---:|
| Random baseline | 24.9 | 10.2% |
| Approximate Q-Learning | 71.0 | 29.1% |
| Feature-based Deep Q-Learning | 103.9 | 42.6% |
| CNN-based DQN | 145.2 | 59.5% |

Reported mean episode duration was 14.6 seconds for the random baseline, 20.0 seconds for AQL, 17.9 seconds for feature-based Deep Q-Learning, and 28.3 seconds for CNN-based DQN.

These figures come from the report's evaluation experiment. They should not be treated as a guarantee of performance when training or evaluating a different branch, checkpoint, configuration, or software environment.

The checked-in CNN training summary is a separate run. It reports 2,250 episodes and 210,000 steps, with a mean of 48.7 pellets per episode across the run, 87.0 over the last 50 episodes, a maximum of 158 pellets, and no full-board wins. These statistics are not directly interchangeable with the report's 50-episode evaluation table.

## Limitations and future work

The experiments did not produce an agent that consistently cleared the entire board. Reinforcement learning results can vary with random initialization, exploration settings, reward shaping, training duration, and hardware.

Potential improvements discussed in the project report include:

- Training for more episodes
- Improving the exploration schedule
- Refining reward design and making reward scales more comparable
- Adding better state features to the feature-based agents
- Exploring recurrent architectures, such as CNN + LSTM, to incorporate temporal memory

## Project structure

A typical checkout containing the base game and CNN agent has a structure similar to:

```text
DVA514/
├── Pacman/
│   ├── run.py
│   ├── pacman.py
│   ├── ghosts.py
│   ├── pellets.py
│   ├── fruit.py
│   ├── nodes.py
│   ├── entity.py
│   ├── constants.py
│   ├── maze1.txt
│   ├── maze1_rotation.txt
│   └── spritesheets / font assets
├── Pacman_DQN/                 # present in the CNN-DQN branch
│   ├── run.py
│   ├── dqn_agent.py
│   ├── constants.py
│   ├── eval_run.py
│   ├── eval.py
│   ├── checkpoints/
│   └── results files
└── Project Report.pdf           
```

The feature-based RL branches add files such as `agentPac.py`, or `env.py`, `DQN.py`, `train_dqn.py`, `run_dqn.py`, and `test_env.py` under `Pacman/`.

## Credits

The project was developed at Mälardalen University, School of Innovation, Design and Engineering, Västerås, Sweden. The project report credits Jonathan Richards' Pygame Pac-Man guide as a reference for the base game implementation.

Contributors listed in the report:

- Kim Svedberg — base Pac-Man customization and CNN-based DQN
- Zebastian Thorsén — feature-based Deep Q-Learning
- Nadja Säfströmer — feature-based Deep Q-Learning
- Antonio Vaca Landeta — Approximate Q-Learning
