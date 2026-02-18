# Pac-Man DQN AI — Deep Q-Learning with TensorFlow CNN

## Overview

This project implements a **Deep Q-Network (DQN)** AI agent to the Pac-Man game.
The agent learns entirely from raw pixel input — no hand-crafted features.

```
pygame screen  →  grayscale 84×84  →  4-frame stack  →  CNN  →  Q-values  →  action
```

## Requirements

```bash
pip install tensorflow pygame numpy
```

Tested with TensorFlow 2.x (CPU or GPU).

---

## Quick Start

### Train the AI

```bash
cd Pacman_DQN
python run.py
```

Set in `constants.py`:
```python
USE_AI         = True   # AI controls Pac-Man
MODEL_TRAINING = True   # no pause screens / ready text (Don't work)
RENDER_MODE    = True   # set False for ~3× faster headless training
```

### Resume from checkpoint

In `run.py`, set:
```python
CHECKPOINT = "checkpoints/dqn_step_50000.weights.h5"
```

### Play yourself (keyboard)

```python
USE_AI = False
```
Arrow keys control Pac-Man.

### Key bindings during AI training

| Key | Action |
|-----|--------|
| `S` | Save checkpoint immediately |
| `Q` | Save and quit |

---

## Architecture

### CNN Q-Network (`dqn_agent.py → build_q_network`)

```
Input  (84, 84, 4)   ← 4 stacked grayscale frames
  Conv2D  32 filters  8×8  stride 4  ReLU
  Conv2D  64 filters  4×4  stride 2  ReLU
  Conv2D  64 filters  3×3  stride 1  ReLU
  Flatten
  Dense   512  ReLU
  Dense   4    Linear   ← Q(s, UP), Q(s, DOWN), Q(s, LEFT), Q(s, RIGHT)
```

Loss: **Huber** (less sensitive to outliers than MSE)  
Optimizer: **Adam** (lr = 0.00005, gradient clip norm = 10)

---

## Algorithm: Double DQN

Standard DQN tends to over-estimate Q-values. We use **Double DQN** to decouple
action *selection* (online network) from action *evaluation* (target network):

```
a* = argmax_a  Q_online(s', a)          # SELECT with online net
y  = r + γ · Q_target(s', a*)           # EVALUATE with target net
loss = Huber(y, Q_online(s, a))
```

The target network weights are hard-copied from the online network every
`TARGET_UPDATE = 2 000` steps.

---

## Frame Preprocessing (`preprocess_frame`)

```
pygame.Surface
  └─ surfarray.array3d()   →  (W, H, 3)  uint8
  └─ transpose             →  (H, W, 3)
  └─ BT.601 grayscale      →  (H, W)     float32
  └─ tf.image.resize       →  (84, 84)   float32   bilinear
  └─ ÷ 255                 →  [0, 1]
```

Four consecutive frames are stacked along the channel axis → `(84, 84, 4)`.
Stacking lets the CNN infer motion (ghost & Pac-Man velocity).

---

## Reward Shaping

| Event           | Reward               |
|-----------------|----------------------|
| Score delta     | `+score_delta × 0.1` |
| Every time-step | `-0.5`               |
| Timeout         | `-300`               |
| Pac-Man dies    | `-400`               |
| Eat pellet      | `+10, +20, +30, +50` |
| Eat powerpellet | `+50`                |
| Eat ghost       | `+150`               |
| Eat fruit       | `+200`               |
| Board cleared   | `+2000`              |

---

## Hyper-parameters (edit in `dqn_agent.py`)

| Parameter | Value   | Description |
|-----------|---------|-------------|
| `FRAME_STACK` | 4       | frames per state |
| `REPLAY_BUFFER` | 50 000  | experience replay capacity |
| `BATCH_SIZE` | 32      | SGD mini-batch |
| `GAMMA` | 0.99    | discount factor |
| `EPS_START` | 1.0     | initial exploration |
| `EPS_END` | 0.1     | minimum exploration |
| `EPS_DECAY` | 0.99999 | per-step multiplier |
| `LEARNING_RATE` | 0.00005  | Adam lr |
| `TARGET_UPDATE` | 2 000   | target sync frequency |
| `TRAIN_START` | 5 000   | warm-up steps before training |
| `SAVE_EVERY` | 10 000  | checkpoint interval |

---

## Training Tips

- **First 5 000 steps**: random play fills the replay buffer — no learning yet.
- **~10 000–50 000 steps**: the agent starts avoiding walls and eating nearby pellets.
- **~100 000+ steps**: consistent ghost avoidance and structured pellet eating emerge.
- Use a GPU for ~5–10× faster CNN inference/training.
- Checkpoints saved to `checkpoints/` every 10 000 steps automatically.

---

## Project Structure

```
pacman_project/
├── dqn_agent.py          ← DQN: CNN, replay buffer, agent logic
├── run.py                ← Game loop with AI hooks
├── constants.py          ← USE_AI, RENDER_MODE, MODEL_TRAINING flags
├── pacman.py             ← AI direction injection
├── checkpoints/          ← Auto-created; model weights saved here
├── animation.py
├── entity.py
├── fruit.py
├── ghosts.py
├── modes.py
├── nodes.py
├── pauser.py
├── pellets.py
├── sprites.py
├── text.py
├── vector.py
├── maze1.txt
├── maze1_rotation.txt
└── spritesheet(new).png
```
