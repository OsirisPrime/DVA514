from run import GameController
from env import PacmanEnv
import numpy as np

game = GameController()
env = PacmanEnv(game)

s = env.reset()
print("state_dim:", env.state_dim())

for i in range(5000):
    a = np.random.randint(env.n_actions)
    s, r, done, info = env.step(a)
    print(i, "a", a, "r", r, "done", done, "info", info)
    if done:
        break