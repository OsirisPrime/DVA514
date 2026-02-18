"""
DQN Agent for Pac-Man using TensorFlow CNN + Deep Q-Learning.

Architecture:
  - Input: Stack of 4 grayscale frames (84x84) captured via pygame screenshot
  - CNN: 3 convolutional layers → Flatten → 2 Dense layers
  - Output: Q-values for 4 actions (UP, DOWN, LEFT, RIGHT)

Training loop uses:
  - Experience Replay (deque-based ReplayBuffer)
  - Target Network (hard-updated every C steps)
  - Epsilon-greedy exploration with exponential decay
"""

import numpy as np
import random
import os
import json
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ─────────────────────────────────────────────
#  Hyper-parameters
# ─────────────────────────────────────────────
FRAME_HEIGHT   = 84          # resize target height
FRAME_WIDTH    = 84          # resize target width
FRAME_STACK    = 4           # number of stacked frames fed to CNN

REPLAY_BUFFER  = 40_000      # max transitions stored
BATCH_SIZE     = 32          # mini-batch size for gradient update
GAMMA          = 0.99        # discount factor

FRAME_SKIP     = 4           # repeat each action for N frames before new decision
                             # reduces CNN inference cost, smooths movement

EPS_START      = 1.0         # initial epsilon (100% random)
EPS_END        = 0.1         # minimum epsilon
EPS_DECAY      = 0.99999     # multiplicative decay per step

LEARNING_RATE  = 0.00005     # Adam learning rate
TARGET_UPDATE  = 2_000       # steps between target-network hard copy
TRAIN_START    = 5_000       # steps before training begins

SAVE_DIR       = "checkpoints"
SAVE_EVERY     = 10_000      # steps between model saves
BUFFER_SAVE_SIZE = 5_000     # transitions to persist with each checkpoint                          


# ─────────────────────────────────────────────
#  CNN Q-Network
# ─────────────────────────────────────────────
def build_q_network(n_actions: int) -> tf.keras.Model:
    """
    Nature-DQN style CNN.
    Input shape: (84, 84, FRAME_STACK)
    """
    inputs = layers.Input(shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_STACK),
                          name="frames")

    x = layers.Conv2D(32, kernel_size=8, strides=4, activation="relu",
                      name="conv1")(inputs)
    x = layers.Conv2D(64, kernel_size=4, strides=2, activation="relu",
                      name="conv2")(x)
    x = layers.Conv2D(64, kernel_size=3, strides=1, activation="relu",
                      name="conv3")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    q_values = layers.Dense(n_actions, activation="linear", name="q_out")(x)

    model = models.Model(inputs=inputs, outputs=q_values, name="DQN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE,
                                  clipnorm=10.0),
        loss="huber"        # Huber is more stable than MSE
    )
    return model


# ─────────────────────────────────────────────
#  Experience Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_BUFFER):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a (s, a, r, s', done) transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, path: str, n: int = BUFFER_SAVE_SIZE):
        """
        Save the last `n` transitions to `path` as a compressed .npz file.
        """
        recent = list(self.buffer)[-n:]
        if not recent:
            return

        states, actions, rewards, next_states, dones = zip(*recent)

        np.savez_compressed(
            path,
            states      = np.array(states,      dtype=np.float16),
            actions     = np.array(actions,      dtype=np.int8),
            rewards     = np.array(rewards,      dtype=np.float32),
            next_states = np.array(next_states,  dtype=np.float16),
            dones       = np.array(dones,        dtype=np.bool_),
        )
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"[DQN] Saved buffer  → {path}  "
              f"({len(recent)} transitions, {size_mb:.1f} MB)")

    def load_buffer(self, path: str):
        """
        Load transitions from a .npz file and push them into the buffer, called on resume.
        """
        data        = np.load(path)
        states      = data["states"].astype(np.float32)
        actions     = data["actions"].astype(np.int32)
        rewards     = data["rewards"].astype(np.float32)
        next_states = data["next_states"].astype(np.float32)
        dones       = data["dones"].astype(np.float32)

        for i in range(len(states)):
            self.push(states[i], int(actions[i]), float(rewards[i]),
                      next_states[i], float(dones[i]))

        print(f"[DQN] Loaded buffer ← {path}  ({len(states)} transitions)")


# ─────────────────────────────────────────────
#  Frame Pre-processing
# ─────────────────────────────────────────────
def preprocess_frame(pygame_surface) -> np.ndarray:
    """
    Convert a pygame Surface → 84×84 grayscale float32 in [0, 1].

    Steps:
      1. pygame.surfarray.array3d  →  (W, H, 3)  uint8  RGB
      2. Transpose                 →  (H, W, 3)
      3. Weighted grayscale        →  (H, W)      float32
      4. Resize via TF             →  (84, 84)    float32
      5. Normalise to [0, 1]
    """
    import pygame

    # Step 1-2: surface → numpy HWC
    rgb = pygame.surfarray.array3d(pygame_surface)                  # (W, H, 3)
    rgb = rgb.transpose(1, 0, 2)                                    # (H, W, 3)

    # Step 3: luminance grayscale
    gray = (0.299 * rgb[:, :, 0] +
            0.587 * rgb[:, :, 1] +
            0.114 * rgb[:, :, 2]).astype(np.float32)                # (H, W)

    # Step 4: resize to 84×84 using TensorFlow bilinear interpolation
    gray_tensor = tf.constant(gray[np.newaxis, :, :, np.newaxis])   # (1,H,W,1)
    resized = tf.image.resize(gray_tensor,
                               [FRAME_HEIGHT, FRAME_WIDTH],
                               method="bilinear")                   # (1,84,84,1)
    gray84 = resized.numpy()[0, :, :, 0]                            # (84,84)

    # Step 5: normalise
    gray84 /= 255.0
    return gray84                                                   # (84,84) float32


# ─────────────────────────────────────────────
#  Frame Stack
# ─────────────────────────────────────────────
class FrameStack:
    """Maintains a rolling window of the last FRAME_STACK grayscale frames."""

    def __init__(self, n: int = FRAME_STACK):
        self.n = n
        self.frames = deque(maxlen=n)

    def reset(self, frame: np.ndarray):
        """Fill stack with copies of the initial frame."""
        for _ in range(self.n):
            self.frames.append(frame)
        return self._get_state()

    def step(self, frame: np.ndarray):
        """Push new frame, return stacked state."""
        self.frames.append(frame)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Return (84, 84, FRAME_STACK) array."""
        return np.stack(self.frames, axis=-1)   # (H, W, N)


# ─────────────────────────────────────────────
#  DQN Agent
# ─────────────────────────────────────────────
class DQNAgent:
    """
    Deep Q-Network agent.

    Action mapping (from constants.py):
        0 → UP    ( 1)
        1 → DOWN  (-1)
        2 → LEFT  ( 2)
        3 → RIGHT (-2)
    """

    ACTION_MAP = {0: 1, 1: -1, 2: 2, 3: -2}
    N_ACTIONS  = 4

    def __init__(self, load_checkpoint: str | None = None):
        self.epsilon    = EPS_START
        self.step_count = 0
        self.episode    = 0

        # Networks
        self.online_net = build_q_network(self.N_ACTIONS)
        self.target_net = build_q_network(self.N_ACTIONS)
        self._sync_target()

        # Memory & stacking
        self.memory      = ReplayBuffer()
        self.frame_stack = FrameStack()

        # ── Per-episode stats (all lists are parallel — index = episode) ──
        self.episode_reward   = 0.0       # accumulator for current episode
        self.episode_rewards  = []        # total shaped reward per episode
        self.episode_scores   = []        # raw game score per episode
        self.episode_pellets  = []        # pellets eaten per episode
        self.episode_steps    = []        # decision steps per episode
        self.episode_outcomes = []        # 'win' | 'death' | 'timeout'
        self.episode_epsilons = []        # epsilon at end of each episode

        # ── Per-training-step loss (downsampled every 10 steps) ───────────
        self.loss_steps  = []             # global step index when loss was recorded
        self.losses      = []             # corresponding loss values

        # ── Warm-up bypass ────────────────────────────────────────────────
        # When starting fresh, the replay buffer must accumulate TRAIN_START
        # transitions before training begins.  When resuming from a checkpoint
        # the model is already trained, so we skip the warm-up immediately
        self._warm_up_done = False

        os.makedirs(SAVE_DIR, exist_ok=True)

        if load_checkpoint:
            self.load(load_checkpoint)

    # ── Network utilities ─────────────────────
    def _sync_target(self):
        """Hard-copy online weights → target network."""
        self.target_net.set_weights(self.online_net.get_weights())

    # ── Epsilon-greedy action selection ───────
    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy policy.
        state : (84, 84, 4)  float32
        returns: integer action index in [0, 3]
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.N_ACTIONS - 1)

        # Greedy: pick argmax Q(s, ·)
        state_t = tf.constant(state[np.newaxis], dtype=tf.float32)  # (1,84,84,4)
        q_vals  = self.online_net(state_t, training=False).numpy()[0]
        return int(np.argmax(q_vals))

    def action_to_direction(self, action_idx: int) -> int:
        """Map agent action index to Pac-Man direction constant."""
        return self.ACTION_MAP[action_idx]

    # ── Training step ─────────────────────────
    def train_step(self):
        """Sample a mini-batch and perform one gradient update."""
        # On a fresh run: wait until the buffer has TRAIN_START entries.
        # On a resumed run: _warm_up_done is set True by load(), so we skip
        # the long wait — but we still need at least BATCH_SIZE transitions
        # before we can sample a full mini-batch.
        if not self._warm_up_done:
            if len(self.memory) < TRAIN_START:
                return None
            self._warm_up_done = True
            print(f"[DQN] Warm-up complete ({TRAIN_START} steps) — training started.")

        if len(self.memory) < BATCH_SIZE:
            return None  # buffer not yet large enough to sample from

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # ── Compute target Q-values (Double DQN) ──────────────────
        # Use online net to SELECT the best action in next state
        next_q_online = self.online_net(
            tf.constant(next_states, dtype=tf.float32), training=False
        ).numpy()                                                       # (B, 4)
        best_actions = np.argmax(next_q_online, axis=1)                 # (B,)

        # Use target net to EVALUATE that action's value
        next_q_target = self.target_net(
            tf.constant(next_states, dtype=tf.float32), training=False
        ).numpy()                                                       # (B, 4)
        next_q = next_q_target[np.arange(BATCH_SIZE), best_actions]     # (B,)

        # Bellman target
        targets = rewards + GAMMA * next_q * (1.0 - dones)              # (B,)

        # ── Build full Q-value matrix; only update chosen actions ────────
        current_q = self.online_net(
            tf.constant(states, dtype=tf.float32), training=False
        ).numpy()                                                       # (B, 4)
        current_q[np.arange(BATCH_SIZE), actions] = targets

        # ── Gradient step ────────────────────────────────────────────────
        history = self.online_net.fit(
            tf.constant(states, dtype=tf.float32),
            tf.constant(current_q, dtype=tf.float32),
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=0
        )
        loss = history.history["loss"][0]
        self.losses.append(loss)
        return loss

    # ── Per-step bookkeeping ──────────────────
    def post_step(self, state, action, reward, next_state, done):
        """
        Call once per game step with:
          state, next_state : (84,84,4) float32
          action            : int index
          reward            : float
          done              : bool
        """
        self.memory.push(state, action, reward, next_state, done)
        self.episode_reward += reward
        self.step_count     += 1

        # Decay epsilon
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

        # Periodic target-network sync
        if self.step_count % TARGET_UPDATE == 0:
            self._sync_target()

        # Periodic checkpoint
        if self.step_count % SAVE_EVERY == 0:
            self.save(f"{SAVE_DIR}/dqn_step_{self.step_count}.weights.h5")

        # Train and record loss every 10 steps
        loss = self.train_step()
        if loss is not None and self.step_count % 10 == 0:
            self.loss_steps.append(self.step_count)
            self.losses.append(float(loss))
        return loss

    def end_episode(self, score: int = 0, pellets: int = 0,
                    steps: int = 0, outcome: str = "death"):
        """
        Call when a game episode ends.

        Parameters
        ----------
        score   : raw in-game score for this episode
        pellets : number of pellets eaten
        steps   : number of agent decision steps taken
        outcome : 'win' | 'death' | 'timeout'
        """
        self.episode_rewards.append(self.episode_reward)
        self.episode_scores.append(score)
        self.episode_pellets.append(pellets)
        self.episode_steps.append(steps)
        self.episode_outcomes.append(outcome)
        self.episode_epsilons.append(self.epsilon)

        ep  = self.episode
        r   = self.episode_reward
        avg = np.mean(self.episode_rewards[-50:])
        print(f"Episode {ep:5d} | Reward: {r:8.1f} | Avg50: {avg:8.1f} | "
              f"Score: {score:6d} | Pellets: {pellets:3d} | "
              f"ε: {self.epsilon:.4f} | Outcome: {outcome} | "
              f"Steps: {self.step_count:,}")

        self.episode_reward = 0.0
        self.episode       += 1

    # ── Persistence ───────────────────────────
    def save(self, path: str):
        """
        Save model weights to `path` (.h5), training state to `path.meta.json`,
        and the last BUFFER_SAVE_SIZE transitions to `path.buffer.npz`.
        """
        self.online_net.save_weights(path)

        meta = {
            "epsilon":          self.epsilon,
            "step_count":       self.step_count,
            "episode":          self.episode,
            # per-episode lists
            "episode_rewards":  self.episode_rewards,
            "episode_scores":   self.episode_scores,
            "episode_pellets":  self.episode_pellets,
            "episode_steps":    self.episode_steps,
            "episode_outcomes": self.episode_outcomes,
            "episode_epsilons": self.episode_epsilons,
            # training loss (downsampled)
            "loss_steps":       self.loss_steps,
            "losses":           self.losses,
        }
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Save replay buffer — last BUFFER_SAVE_SIZE transitions
        buf_path = path + ".buffer.npz"
        self.memory.save_buffer(buf_path, n=BUFFER_SAVE_SIZE)

        print(f"[DQN] Saved weights  → {path}")
        print(f"[DQN] Saved metadata → {meta_path}  "
              f"(ε={self.epsilon:.4f}, steps={self.step_count:,}, "
              f"episode={self.episode})")

    def load(self, path: str):
        """
        Load model weights, restore training state from `.meta.json`, and
        pre-fill the replay buffer from `.buffer.npz` if it exists.
        """
        self.online_net.load_weights(path)
        self._sync_target()
        print(f"[DQN] Loaded weights ← {path}")

        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

            self.epsilon          = meta.get("epsilon",          EPS_END)
            self.step_count       = meta.get("step_count",       0)
            self.episode          = meta.get("episode",          0)
            self.episode_rewards  = meta.get("episode_rewards",  [])
            self.episode_scores   = meta.get("episode_scores",   [])
            self.episode_pellets  = meta.get("episode_pellets",  [])
            self.episode_steps    = meta.get("episode_steps",    [])
            self.episode_outcomes = meta.get("episode_outcomes", [])
            self.episode_epsilons = meta.get("episode_epsilons", [])
            self.loss_steps       = meta.get("loss_steps",       [])
            self.losses           = meta.get("losses",           [])

            print(f"[DQN] Restored metadata ← {meta_path}  "
                  f"(ε={self.epsilon:.4f}, steps={self.step_count:,}, "
                  f"episode={self.episode})")
        else:
            print(f"[DQN] No metadata file at {meta_path} — "
                  f"epsilon/steps/episode start from defaults.")

        # Pre-fill replay buffer from saved transitions
        buf_path = path + ".buffer.npz"
        if os.path.exists(buf_path):
            self.memory.load_buffer(buf_path)
        else:
            print(f"[DQN] No buffer file at {buf_path} — "
                  f"buffer starts empty (first {BATCH_SIZE} steps skipped).")

        # Skip the warm-up — model is already trained and buffer is pre-filled
        self._warm_up_done = True
        print(f"[DQN] Warm-up bypassed — training resumes immediately.")

