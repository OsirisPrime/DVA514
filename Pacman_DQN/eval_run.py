"""
eval_run.py  —  Final Evaluation of Trained DQN Pac-Man Agent
==============================================================
Loads the latest (or specified) checkpoint and runs the agent in pure
greedy mode (ε = 0, no exploration, no training) for a fixed number of
episodes.  Results are saved to eval_results/ and a summary plot is produced.

Key differences from run.py (training mode)
--------------------------------------------
  - ε = 0 always  →  agent always picks the highest Q-value action
  - No replay buffer, no gradient updates, no epsilon decay
  - Runs exactly N_EVAL_EPISODES then exits automatically
  - Separate output folder (eval_results/) so training data is untouched

Usage
-----
    python eval_run.py                        # auto-loads latest checkpoint
    python eval_run.py --checkpoint path.h5   # specific checkpoint
    python eval_run.py --episodes 20          # change number of eval episodes
    python eval_run.py --no-render            # headless (faster)

Output (in eval_results/)
------
    eval_results.json   — raw per-episode data
    eval_summary.txt    — printed statistics table
    eval_plots.png      — performance plots (300 DPI)
    eval_plots.pdf      — vector version for LaTeX
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

import pygame
from pygame.locals import *

# ── Game imports ─────────────────────────────────────────────────────────────
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites, MazeSprites
from dqn_agent import preprocess_frame, build_q_network, FRAME_SKIP

import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_EVAL_EPISODES = 10          # episodes to run before auto-exit
EVAL_TIMEOUT    = 300         # 5 min per episode (shorter than training)
EVAL_FPS        = 30          # display FPS
RESULTS_DIR     = "eval_results"

CHECKPOINT_DIR  = "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(ckpt_dir: str = CHECKPOINT_DIR) -> str | None:
    """Return the .h5 path with the highest step number, or None."""
    p = Path(ckpt_dir)
    if not p.exists():
        return None
    files = [f for f in p.glob("*.weights.h5")
             if not f.name.startswith("dqn_quit")]
    if not files:
        return None
    return str(sorted(
        files,
        key=lambda f: int("".join(filter(str.isdigit, f.stem)) or "0")
    )[-1])


# ─────────────────────────────────────────────────────────────────────────────
#  Greedy Agent  (inference only — no training infrastructure)
# ─────────────────────────────────────────────────────────────────────────────
class GreedyAgent:
    """
    Thin wrapper around the CNN Q-network for pure greedy inference.
    No replay buffer, no target network, no epsilon, no gradient updates.
    """
    ACTION_MAP = {0: 1, 1: -1, 2: 2, 3: -2}
    N_ACTIONS  = 4

    def __init__(self, checkpoint_path: str):
        self.net = build_q_network(self.N_ACTIONS)
        self.net.load_weights(checkpoint_path)
        print(f"[Eval] Loaded model ← {checkpoint_path}")

        # Read metadata for display purposes only
        meta_path = checkpoint_path + ".meta.json"
        self.train_steps   = 0
        self.train_episode = 0
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.train_steps   = meta.get("step_count", 0)
            self.train_episode = meta.get("episode",    0)
            print(f"[Eval] Checkpoint trained for {self.train_steps:,} steps "
                  f"over {self.train_episode} episodes")

        # Frame stack (4 frames, same as training)
        from dqn_agent import FrameStack
        self.frame_stack = FrameStack()
        self._init_stack = True

    def reset(self):
        """Call at the start of each episode to flush stale frames."""
        self._init_stack = True

    def act(self, screen) -> int:
        """
        Capture screen, update frame stack, return greedy action index.
        ε = 0 — always picks argmax Q(s, ·).
        """
        frame = preprocess_frame(screen)

        if self._init_stack:
            state = self.frame_stack.reset(frame)
            self._init_stack = False
        else:
            state = self.frame_stack.step(frame)

        state_t = tf.constant(state[np.newaxis], dtype=tf.float32)
        q_vals  = self.net(state_t, training=False).numpy()[0]
        return int(np.argmax(q_vals))

    def action_to_direction(self, action_idx: int) -> int:
        return self.ACTION_MAP[action_idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation Game Controller
# ─────────────────────────────────────────────────────────────────────────────
class EvalController:

    def __init__(self, agent: GreedyAgent, n_episodes: int, render: bool):
        pygame.init()
        self.screen      = pygame.display.set_mode(SCREENSIZE, 0, 32)
        pygame.display.set_caption("Pac-Man DQN — Evaluation")
        self.background  = None
        self.clock       = pygame.time.Clock()
        self.fruit       = None
        self.level       = 0
        self.lives       = NUMLIVES
        self.score       = 0
        self.textgroup   = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.pause       = Pause(False)
        self.render_mode = render

        self.agent      = agent
        self.n_episodes = n_episodes

        # Per-episode tracking
        self.episodes_done   = 0
        self.results         = []   # list of dicts, one per episode

        # Within-episode state
        self._skip_counter   = 0
        self._episode_start  = time.time()
        self._episode_score  = 0
        self._episode_steps  = 0
        self._outcome        = "death"

    # ─────────────────────────────────────────
    #  Game lifecycle
    # ─────────────────────────────────────────
    def _record_and_restart(self, outcome: str):
        """Save episode result and either restart or exit."""
        elapsed = time.time() - self._episode_start
        result  = {
            "episode":  self.episodes_done + 1,
            "score":    self._episode_score,
            "pellets":  self.pellets.numEaten,
            "steps":    self._episode_steps,
            "outcome":  outcome,
            "duration": round(elapsed, 2),
        }
        self.results.append(result)

        ep = result["episode"]
        print(f"  Episode {ep:3d}/{self.n_episodes}  |  "
              f"Score: {result['score']:5d}  |  "
              f"Pellets: {result['pellets']:3d}/240  |  "
              f"Steps: {result['steps']:5d}  |  "
              f"Outcome: {outcome:<8}  |  "
              f"Time: {elapsed:.1f}s")

        self.episodes_done += 1
        if self.episodes_done >= self.n_episodes:
            self._finish()
        else:
            self._reset_episode()
            self.restartGame()

    def _reset_episode(self):
        self.agent.reset()
        self._skip_counter  = 0
        self._episode_start = time.time()
        self._episode_score = 0
        self._episode_steps = 0
        self._outcome       = "death"

    def restartGame(self):
        self.lives  = NUMLIVES
        self.level  = 0
        self.score  = 0
        self.fruit  = None
        self.pause.paused = False
        self.startGame()
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.lifesprites.resetLives(self.lives)

    def resetLevel(self):
        """Called on life loss (lives > 0)."""
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        # Reset frame stack so CNN doesn't see stale pre-death frames
        self.agent._init_stack = True
        self._skip_counter = 0

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)

    def startGame(self):
        self.setBackground()
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.background  = self.mazesprites.constructBackground(
            self.background, self.level % 5)
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0, 17), (27, 17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12, 14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15, 14), RIGHT)
        self.pacman  = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts  = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2  + 11.5, 0 + 14))
        self.ghosts.pinky.setStartNode( self.nodes.getNodeFromTiles(2  + 11.5, 3 + 14))
        self.ghosts.inky.setStartNode(  self.nodes.getNodeFromTiles(0  + 11.5, 3 + 14))
        self.ghosts.clyde.setStartNode( self.nodes.getNodeFromTiles(4  + 11.5, 3 + 14))
        self.ghosts.setSpawnNode(       self.nodes.getNodeFromTiles(2  + 11.5, 3 + 14))
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2 + 11.5, 3 + 14, LEFT,  self.ghosts)
        self.nodes.denyAccessList(2 + 11.5, 3 + 14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess( RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT,  self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)

    # ─────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────
    def update(self):
        dt = self.clock.tick(EVAL_FPS) / 1000.0
        dt = min(dt, 2.0 / EVAL_FPS)   # cap to prevent tunnelling
        self.textgroup.update(dt)
        self.pellets.update(dt)

        if not self.pause.paused:
            # ── Timeout ──────────────────────────────────────────────
            if time.time() - self._episode_start >= EVAL_TIMEOUT:
                print(f"  [timeout after {EVAL_TIMEOUT}s]")
                self._record_and_restart("timeout")
                return

            # ── Agent decision every FRAME_SKIP frames ───────────────
            if self._skip_counter == 0:
                action = self.agent.act(self.screen)
                self.pacman.ai_direction = self.agent.action_to_direction(action)
                self._episode_steps += 1

            self.pacman.update(dt)
            self.ghosts.update(dt)
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()

        self._skip_counter = (self._skip_counter + 1) % FRAME_SKIP
        self.checkEvents()
        self.render()

    # ─────────────────────────────────────────
    #  Events
    # ─────────────────────────────────────────
    def updateScore(self, points):
        self.score        += points
        self._episode_score = self.score
        self.textgroup.updateScore(self.score)

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self._finish()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self._finish()
                elif event.key == K_SPACE:
                    # Allow manual pause during eval for observation
                    self.pause.setPause(playerPaused=True)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    self.updateScore(ghost.points)
                    self.textgroup.addText(
                        str(ghost.points), WHITE,
                        ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    ghost.visible = False
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -= 1
                        self.lifesprites.removeImage()
                        self.pacman.die()
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self._record_and_restart("death")
                        else:
                            self.pause.setPause(pauseTime=2, func=self.resetLevel)

    def checkFruitEvents(self):
        if self.pellets.numEaten in (50, 140):
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20))
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(
                    str(self.fruit.points), WHITE,
                    self.fruit.position.x, self.fruit.position.y, 8, time=1)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self._record_and_restart("win")

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def render(self):
        if self.render_mode:
            self.screen.blit(self.background, (0, 0))
            self.pellets.render(self.screen)
            if self.fruit is not None:
                self.fruit.render(self.screen)
            self.pacman.render(self.screen)
            self.ghosts.render(self.screen)
            self.textgroup.render(self.screen)
            for i, img in enumerate(self.lifesprites.images):
                x = img.get_width() * i
                y = SCREENHEIGHT - img.get_height()
                self.screen.blit(img, (x, y))
            pygame.display.update()

    # ─────────────────────────────────────────
    #  Exit
    # ─────────────────────────────────────────
    def _finish(self):
        pygame.quit()
        save_and_plot(self.results, self.agent)
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
#  Results: save JSON + summary text + plots
# ─────────────────────────────────────────────────────────────────────────────
def save_and_plot(results: list[dict], agent: GreedyAgent):
    if not results:
        print("[Eval] No results to save.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "train_steps":   agent.train_steps,
            "train_episode": agent.train_episode,
            "n_episodes":    len(results),
            "episodes":      results,
        }, f, indent=2)
    print(f"\n[Eval] Raw data  → {json_path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    scores   = [r["score"]    for r in results]
    pellets  = [r["pellets"]  for r in results]
    steps    = [r["steps"]    for r in results]
    durations= [r["duration"] for r in results]
    outcomes = [r["outcome"]  for r in results]

    wins     = outcomes.count("win")
    deaths   = outcomes.count("death")
    timeouts = outcomes.count("timeout")
    n        = len(results)

    lines = [
        "=" * 56,
        "  DQN PAC-MAN  —  EVALUATION RESULTS",
        "=" * 56,
        f"  Checkpoint trained : {agent.train_steps:,} steps  |  {agent.train_episode} episodes",
        f"  Evaluation episodes: {n}",
        f"  Epsilon (greedy)   : 0.000  (no exploration)",
        "",
        "  Game score",
        f"    Mean             : {np.mean(scores):.1f}",
        f"    Std              : {np.std(scores):.1f}",
        f"    Min / Max        : {min(scores)} / {max(scores)}",
        "",
        "  Pellets eaten / episode  (max 240)",
        f"    Mean             : {np.mean(pellets):.1f}",
        f"    Std              : {np.std(pellets):.1f}",
        f"    Min / Max        : {min(pellets)} / {max(pellets)}",
        "",
        "  Episode length (decision steps)",
        f"    Mean             : {np.mean(steps):.1f}",
        f"    Min / Max        : {min(steps)} / {max(steps)}",
        "",
        "  Episode duration (seconds)",
        f"    Mean             : {np.mean(durations):.1f}s",
        f"    Min / Max        : {min(durations):.1f}s / {max(durations):.1f}s",
        "",
        "  Outcomes",
        f"    Wins             : {wins}  ({100*wins/n:.1f}%)",
        f"    Deaths           : {deaths}  ({100*deaths/n:.1f}%)",
        f"    Timeouts         : {timeouts}  ({100*timeouts/n:.1f}%)",
        "=" * 56,
    ]
    summary = "\n".join(lines)
    print("\n" + summary)

    txt_path = os.path.join(RESULTS_DIR, "eval_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"[Eval] Summary   → {txt_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        import matplotlib.ticker
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        ep_nums = [r["episode"] for r in results]

        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor("white")
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.45, wspace=0.38,
                                left=0.07, right=0.97,
                                top=0.90,  bottom=0.08)

        COLORS = {
            "score":   "#4CAF50",
            "pellets": "#9C27B0",
            "steps":   "#00BCD4",
            "dur":     "#FF9800",
            "bar_w":   "#2196F3",
            "bar_d":   "#F44336",
            "bar_t":   "#FF9800",
        }

        def style(ax, title, xlabel, ylabel):
            ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, linewidth=0.6, alpha=0.4, linestyle="--")

        def bar_with_mean(ax, x, y, color, ylabel):
            ax.bar(x, y, color=color, alpha=0.7, width=0.7)
            mean_val = np.mean(y)
            ax.axhline(mean_val, color="#212121", linewidth=2.0,
                       linestyle="--", label=f"Mean: {mean_val:.1f}")
            ax.legend(fontsize=7, framealpha=0.7)

        # ── Score per episode ────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        bar_with_mean(ax, ep_nums, scores, COLORS["score"], "Score")
        style(ax, "Score per Episode", "Episode", "Game score")

        # ── Pellets per episode ──────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        bar_with_mean(ax, ep_nums, pellets, COLORS["pellets"], "Pellets")
        ax.axhline(240, color="#aaa", linewidth=1.5, linestyle=":",
                   label="Board (240)")
        ax.set_ylim(0, 255)
        ax.legend(fontsize=7, framealpha=0.7)
        style(ax, "Pellets Eaten per Episode", "Episode", "Pellets")

        # ── Outcomes pie ─────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 2])
        outcome_counts = {k: v for k, v in
                          [("Win", wins), ("Death", deaths), ("Timeout", timeouts)]
                          if v > 0}
        pie_colors = ["#4CAF50", "#F44336", "#FF9800"][:len(outcome_counts)]
        wedges, texts, autotexts = ax.pie(
            outcome_counts.values(),
            labels=outcome_counts.keys(),
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 8},
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        )
        for t in autotexts:
            t.set_fontsize(7)
        ax.set_title("Outcome Distribution", fontsize=10,
                     fontweight="bold", pad=5)

        # ── Episode steps ────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        bar_with_mean(ax, ep_nums, steps, COLORS["steps"], "Steps")
        style(ax, "Episode Length (decision steps)", "Episode", "Steps")

        # ── Episode duration ─────────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        bar_with_mean(ax, ep_nums, durations, COLORS["dur"], "Seconds")
        style(ax, "Episode Duration (seconds)", "Episode", "Seconds")

        # ── Score distribution histogram ─────────────────────────────────────
        ax = fig.add_subplot(gs[1, 2])
        ax.hist(scores, bins=min(n, 15), color=COLORS["score"],
                alpha=0.75, edgecolor="white", linewidth=0.8)
        ax.axvline(np.mean(scores), color="#212121", linewidth=2.0,
                   linestyle="--", label=f"Mean: {np.mean(scores):.0f}")
        ax.legend(fontsize=7, framealpha=0.7)
        style(ax, "Score Distribution", "Score", "Count")

        fig.suptitle(
            f"Pac-Man DQN — Evaluation  "
            f"({n} episodes  |  trained {agent.train_steps:,} steps  |  ε = 0.000)",
            fontsize=12, fontweight="bold", y=0.96
        )

        png_path = os.path.join(RESULTS_DIR, "eval_plots.png")
        pdf_path = os.path.join(RESULTS_DIR, "eval_plots.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Eval] Plots     → {png_path}")
        print(f"[Eval] Plots     → {pdf_path}")

    except Exception as e:
        print(f"[Eval] Plotting skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI + entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Run final greedy evaluation of a trained DQN Pac-Man agent")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .h5 checkpoint (default: auto-detect latest)")
    p.add_argument("--episodes", type=int, default=N_EVAL_EPISODES,
                   help=f"Number of evaluation episodes (default: {N_EVAL_EPISODES})")
    p.add_argument("--no-render", action="store_true",
                   help="Run headless without a display window (faster)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Find checkpoint ───────────────────────────────────────────────────────
    ckpt = args.checkpoint or find_latest_checkpoint()
    if ckpt is None:
        print(f"[Eval] ERROR: no checkpoint found in {CHECKPOINT_DIR}/")
        print("       Train first with run.py or pass --checkpoint <path>")
        sys.exit(1)

    render = not args.no_render

    # ── Load agent ────────────────────────────────────────────────────────────
    agent = GreedyAgent(ckpt)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"\n[Eval] Running {args.episodes} greedy evaluation episodes "
          f"(ε = 0.000, no training)...")
    print(f"       Press ESC or close the window to stop early.\n")

    game = EvalController(agent, n_episodes=args.episodes, render=render)
    game.startGame()

    while True:
        game.update()


if __name__ == "__main__":
    main()
