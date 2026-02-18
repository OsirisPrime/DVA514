# train_dqn.py
import os
import numpy as np
from run import GameController
from env import PacmanEnv
from DQN import DQNAgent

MODEL_PATH = "dqn_pacman.weights.h5"
META_PATH  = "dqn_meta.npz"

def load_or_create_agent(state_dim, n_actions):
    agent = DQNAgent(state_dim, n_actions=n_actions)

    # Ladda modell om den finns
    if os.path.exists(MODEL_PATH):
        agent.q.load_weights(MODEL_PATH)
        agent.q_target.set_weights(agent.q.get_weights())
        print(f"Loaded model weights from {MODEL_PATH}")

    # Ladda epsilon/step om du vill fortsätta snyggt
    if os.path.exists(META_PATH):
        meta = np.load(META_PATH)
        if "eps" in meta:
            agent.eps = float(meta["eps"])
        if "global_step" in meta:
            agent.global_step = int(meta["global_step"])
        print(f"Loaded meta from {META_PATH}: eps={agent.eps:.3f}, step={getattr(agent,'global_step',0)}")
    else:
        agent.global_step = 0

    return agent

def save_agent(agent):
    agent.q.save_weights(MODEL_PATH)
    np.savez(META_PATH,
             eps=np.array(agent.eps, dtype=np.float32),
             global_step=np.array(getattr(agent, "global_step", 0), dtype=np.int64))
    print(f"Saved model to {MODEL_PATH} and meta to {META_PATH}")

def train(num_episodes=1000):
    game = GameController()
    env = PacmanEnv(game)

    s = env.reset()
    state_dim = env.state_dim()
    n_actions = env.n_actions
    agent = load_or_create_agent(state_dim, n_actions)

    batch_size = 64
    warmup_steps = 2000
    target_update_every = 1000

    for ep in range(num_episodes):
        s = env.reset()
        ep_reward = 0.0

        stuck_steps = 0
        last_pos = None

        while True:
            a = agent.act(s)
            s2, r, done, info = env.step(a)

            agent.rb.add(s, a, r, s2, done)
            s = s2
            ep_reward += r
            agent.global_step += 1

            # stuck-detektering (om pacman inte rör sig)
            p = env.game.pacman.position
            pos = (float(p.x), float(p.y))
            if last_pos is not None and pos == last_pos:
                stuck_steps += 1
            else:
                stuck_steps = 0
            last_pos = pos

            if stuck_steps > 120:   # ~2 sek
                done = True
                ep_reward -= 50
                info["stuck_reset"] = True

            if len(agent.rb) >= warmup_steps:
                bs, ba, br, bs2, bd = agent.rb.sample(batch_size)
                agent.train_step(bs, ba, br, bs2, bd)

            if agent.global_step % target_update_every == 0:
                agent.update_target()

            if done:
                agent.decay_epsilon()
                break

        print(f"EP {ep+1}/{num_episodes} reward={ep_reward:.1f} eps={agent.eps:.3f} score={info.get('score')} stuck={info.get('stuck_reset', False)}")

        if (ep + 1) % 20 == 0:
            save_agent(agent)

    save_agent(agent)
if __name__ == "__main__":
    train(num_episodes=1000)