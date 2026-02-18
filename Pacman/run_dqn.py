# test_run.py
import numpy as np
from run import GameController
from env import PacmanEnv
from DQN import DQNAgent

MODEL_PATH = "dqn_pacman.weights.h5"

def play_one_episode(max_steps=6000):
    game = GameController()
    env = PacmanEnv(game)

    s = env.reset()
    state_dim = env.state_dim()
    n_actions = env.n_actions

    agent = DQNAgent(state_dim, n_actions=n_actions)
    agent.q.load_weights(MODEL_PATH)
    agent.q_target.set_weights(agent.q.get_weights())

    agent.eps = 0.0  # <-- viktigt: ingen exploration vid test

    total_reward = 0.0
    steps = 0

    while True:
        a = agent.act(s)
        s, r, done, info = env.step(a)
        total_reward += r
        steps += 1

        if done or steps >= max_steps:
            break

    print("DONE:", "steps=", steps, "score=", info.get("score"), "reward=", total_reward)

if __name__ == "__main__":
    play_one_episode()