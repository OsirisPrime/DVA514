import random, collections
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = collections.deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done = map(np.array, zip(*batch))
        return s, a, r.astype(np.float32), s2, done.astype(np.float32)

    def __len__(self):
        return len(self.buf)

def build_q_network(state_dim, n_actions):
    model = keras.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(n_actions)  # Q-values
    ])
    return model

class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99):
        self.n_actions = n_actions
        self.gamma = gamma

        self.q = build_q_network(state_dim, n_actions)
        self.q_target = build_q_network(state_dim, n_actions)
        self.q_target.set_weights(self.q.get_weights())

        self.opt = keras.optimizers.Adam(lr)
        self.rb = ReplayBuffer()

        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        qvals = self.q(np.expand_dims(state, 0), training=False).numpy()[0]
        return int(np.argmax(qvals))

    @tf.function
    def train_step(self, s, a, r, s2, done):
        # Double DQN (stabilare än vanilla): välj action med online-nätet, evaluera med target
        next_actions = tf.argmax(self.q(s2, training=False), axis=1)
        next_q = tf.reduce_sum(
            self.q_target(s2, training=False) * tf.one_hot(next_actions, self.n_actions),
            axis=1
        )
        target = r + (1.0 - done) * self.gamma * next_q

        with tf.GradientTape() as tape:
            q_all = self.q(s, training=True)
            q_sa = tf.reduce_sum(q_all * tf.one_hot(a, self.n_actions), axis=1)
            loss = tf.keras.losses.Huber()(target, q_sa)

        grads = tape.gradient(loss, self.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables))
        return loss

    def update_target(self):
        self.q_target.set_weights(self.q.get_weights())

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)