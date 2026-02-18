"""
run.py  –  Pac-Man + DQN AI (TensorFlow CNN + Deep Q-Learning)

Reward shaping (explicit per-event, not score-delta):
  +10    eat a regular pellet
  +50    eat a power pellet
  +200   eat a frightened ghost  (stacks per ghost)
  +fruit eat a fruit             (uses the fruit's own point value)
  -500   Pac-Man dies
  -200   episode timeout (10 min limit)
  -0.1   per time-step (encourages speed)
  +500   all pellets cleared (win)

Checkpoint resume:
  Set CHECKPOINT in run.py to a .h5 path.
  Epsilon, step count and episode index are restored automatically
  from the companion .meta.json file saved alongside every checkpoint.

Toggle AI / human play:
  USE_AI = True  in constants.py  →  DQN agent
  USE_AI = False                  →  arrow-key control

Key bindings during AI training:
  S    manual checkpoint save
  Q    save and quit
"""

import time
import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites, MazeSprites

# ── Import AI only when enabled ──────────────────────────────────────────────
if USE_AI:
    from dqn_agent import DQNAgent, preprocess_frame, SAVE_DIR, FRAME_SKIP

    # Set to a .h5 path string to resume training, or None to start fresh.
    #CHECKPOINT = None      # e.g. "checkpoints/dqn_step_50000.weights.h5"
    CHECKPOINT = "checkpoints/dqn_step_210000.weights.h5"

    # Maximum seconds per episode
    EPISODE_TIMEOUT = 600   # 10 minutes


# ─────────────────────────────────────────────────────────────────────────────
class GameController(object):

    def __init__(self):
        pygame.init()
        self.screen      = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background  = None
        self.clock       = pygame.time.Clock()
        self.fruit       = None
        self.level       = 0
        self.lives       = NUMLIVES
        self.score       = 0
        self.textgroup   = TextGroup()
        self.lifesprites = LifeSprites(self.lives)

        if MODEL_TRAINING:
            self.pause = Pause(False)
        else:
            self.pause = Pause(True)

        # ── DQN bookkeeping ───────────────────────────────────────────
        if USE_AI:
            self.agent             = DQNAgent(load_checkpoint=CHECKPOINT)
            self._prev_score       = 0
            self._prev_pellets     = 0
            self._decision_state   = None           # state at the moment of last decision
            self._decision_action  = None           # action chosen at last decision
            self._skip_counter     = 0              # frames elapsed since last decision
            self._skip_reward      = 0.0            # reward accumulated over skip window
            self._event_reward     = 0.0            # explicit per-event reward this frame
            self._init_stack_next  = True           # reset frame stack on first act
            self._episode_start    = time.time()    # for 10-min timeout
            self._episode_steps    = 0              # decision steps this episode
            self._episode_outcome  = "death"        # updated on win / timeout

    # ─────────────────────────────────────────
    #  Game lifecycle helpers
    # ─────────────────────────────────────────
    def restartGame(self):
        # Capture episode stats BEFORE resetting score/pellets
        if USE_AI:
            _final_score   = self.score
            _final_pellets = self.pellets.numEaten if hasattr(self, "pellets") else 0
            _final_steps   = self._episode_steps
            _final_outcome = self._episode_outcome

        self.lives  = NUMLIVES
        self.level  = 0
        self.score  = 0
        self.fruit  = None
        if MODEL_TRAINING:
            self.pause.paused = False
        else:
            self.pause.paused = True
        self.startGame()
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        if not MODEL_TRAINING:
            self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        if USE_AI:
            self._prev_score      = 0
            self._prev_pellets    = 0
            self._decision_state  = None
            self._decision_action = None
            self._skip_counter    = 0
            self._skip_reward     = 0.0
            self._event_reward    = 0.0
            self._init_stack_next = True
            self._episode_start   = time.time()
            self._episode_steps   = 0
            self._episode_outcome = "death"
            self.agent.end_episode(
                score   = _final_score,
                pellets = _final_pellets,
                steps   = _final_steps,
                outcome = _final_outcome,
            )

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        if not MODEL_TRAINING:
            self.textgroup.showText(READYTXT)
        if USE_AI:
            # Flush any partial skip window so we don't bleed reward
            # across the life boundary, then reset the frame stack so the
            # CNN doesn't see stale pre-death frames after the respawn.
            self._skip_counter    = 0
            self._skip_reward     = 0.0
            self._event_reward    = 0.0
            self._decision_state  = None
            self._decision_action = None
            self._init_stack_next = True   # rebuild frame stack on next act


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
    #  Main game loop step
    # ─────────────────────────────────────────
    def update(self):
        dt = self.clock.tick(FPS) / 1000.0
        # Cap dt so a slow training step can't make Pac-Man tunnel through
        # entities on the next frame. At speed=100px/s a cap of 2 frames
        # (66ms) limits movement to ~6.6px.
        dt = min(dt, 2.0 / FPS)
        self.textgroup.update(dt)
        self.pellets.update(dt)

        if not self.pause.paused:
            # ── 10-minute episode timeout ─────────────────────────────
            if USE_AI:
                elapsed = time.time() - self._episode_start
                if elapsed >= EPISODE_TIMEOUT:
                    self._episode_outcome  = "timeout"
                    self._event_reward    += TIMEOUT_PENALTY
                    self._ai_step_reward()
                    self._ai_learn()
                    remaining = int(EPISODE_TIMEOUT // 60)
                    print(f"[DQN] ⏱  Episode timed out after "
                          f"{remaining} min — restarting.")
                    self.pause.setPause(pauseTime=0, func=self.restartGame)

            if USE_AI:
                # New decision every FRAME_SKIP frames.
                if self._skip_counter == 0:
                    self._ai_choose_action()
                    self._episode_steps += 1   # count decision steps, not frames

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

        self.checkEvents()
        self.render()

        # After each rendered frame: accumulate reward and maybe learn
        if USE_AI and not self.pause.paused:
            self._ai_step_reward()
            self._skip_counter += 1
            if self._skip_counter >= FRAME_SKIP:
                self._ai_learn()
                self._skip_counter = 0

    # ─────────────────────────────────────────
    #  DQN: choose action (decision frame only)
    # ─────────────────────────────────────────
    def _ai_choose_action(self):
        """
        Capture current frame, build stacked state, run the policy,
        and write the result into pacman.ai_direction.
        """
        frame = preprocess_frame(self.screen)                   # (84, 84) float32

        if self._init_stack_next or self._decision_state is None:
            state = self.agent.frame_stack.reset(frame)
            self._init_stack_next = False
        else:
            state = self.agent.frame_stack.step(frame)

        self._decision_state  = state                           # (84, 84, 4)
        action_idx            = self.agent.select_action(state)
        self._decision_action = action_idx

        # Write requested direction — validated by the node system each tick
        self.pacman.ai_direction = self.agent.action_to_direction(action_idx)
        self._skip_reward = 0.0   # reset accumulator for this skip window

    # ─────────────────────────────────────────
    #  DQN: per-frame reward accumulation
    # ─────────────────────────────────────────
    def _ai_step_reward(self):
        """
        Accumulate reward for each physics frame inside a skip window.
        """
        reward = TIMESTEP_PENALTY           # time-step penalty (encourages speed)

        # Collect explicit per-event rewards set during this frame
        reward += self._event_reward
        self._event_reward = 0.0            # reset for next frame

        # Terminal bonuses (done flag is set in _ai_learn)
        if not self.pacman.alive:
            reward += PACMAN_DIES_PENALTY
        elif self.pellets.isEmpty():
            reward += WIN_REWARD

        self._skip_reward += reward

    # ─────────────────────────────────────────
    #  DQN: end-of-skip store + train
    # ─────────────────────────────────────────
    def _ai_learn(self):
        """
        At the end of a FRAME_SKIP window: capture next state, store the
        (s, a, R, s', done) transition with the accumulated reward, and
        trigger a training step.
        """
        if self._decision_state is None or self._decision_action is None:
            return

        done = not self.pacman.alive or self.pellets.isEmpty()

        # Next state from the current (post-skip) screen
        next_frame = preprocess_frame(self.screen)
        next_state = self.agent.frame_stack.step(next_frame)   # (84,84,4)

        self.agent.post_step(
            self._decision_state,
            self._decision_action,
            self._skip_reward,
            next_state,
            float(done)
        )

    # ─────────────────────────────────────────
    #  Score
    # ─────────────────────────────────────────
    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    # ─────────────────────────────────────────
    #  Events
    # ─────────────────────────────────────────
    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                if USE_AI:
                    self.agent.save(f"{SAVE_DIR}/dqn_quit.weights.h5")
                exit()
            elif event.type == KEYDOWN:
                # Human pause (only in human mode)
                if event.key == K_SPACE and not USE_AI:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            self.hideEntities()
                # Manual checkpoint save
                elif event.key == K_s and USE_AI:
                    self.agent.save(
                        f"{SAVE_DIR}/dqn_manual_{self.agent.step_count}.weights.h5"
                    )
                # Quit + save
                elif event.key == K_q and USE_AI:
                    self.agent.save(f"{SAVE_DIR}/dqn_quit.weights.h5")
                    exit()

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
                    # ── Reward: ate a frightened ghost ────────────────
                    if USE_AI:
                        self._event_reward += EATGHOST_REWARD
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -= 1
                        self.lifesprites.removeImage()
                        self.pacman.die()
                        self.ghosts.hide()
                        if self.lives <= 0:
                            if not MODEL_TRAINING:
                                self.textgroup.showText(GAMEOVERTXT)
                            self.pause.setPause(pauseTime=2, func=self.restartGame)
                        else:
                            self.pause.setPause(pauseTime=2, func=self.resetLevel)

    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20))
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(
                    str(self.fruit.points), WHITE,
                    self.fruit.position.x, self.fruit.position.y, 8, time=1)
                # ── Reward: ate fruit (scales with its point value) ───
                if USE_AI:
                    self._event_reward += FRUIT_REWARD
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
            if USE_AI:
                eaten = self.pellets.numEaten
                if eaten == 200:
                    self._event_reward += 300.0     # bonus for reaching 200
                elif eaten == 150:
                    self._event_reward += 150.0     # bonus for reaching 150
                elif eaten == 100:
                    self._event_reward += 75.0      # bonus for reaching 100
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
                # ── Reward: power pellet ──────────────────────────────
                if USE_AI:
                    self._event_reward += POWERPELLET_REWARD
            else:
                # ── Reward: regular pellet ────────────────────────────
                if USE_AI:
                    # Escalating bonus — more reward the closer to clearing the board
                    eaten = self.pellets.numEaten
                    if eaten >= 200:                        # last 40 pellets
                        self._event_reward += (PELLET_REWARD + 40)
                    elif eaten >= 150:                      # pellets 150-200
                        self._event_reward += (PELLET_REWARD + 20)
                    elif eaten >= 100:                      # pellets 100-150
                        self._event_reward += (PELLET_REWARD + 10)
                    else:
                        self._event_reward += PELLET_REWARD # normal +10
            if self.pellets.isEmpty():
                if USE_AI:
                    self._episode_outcome = "win"
                self.hideEntities()
                self.pause.setPause(pauseTime=2, func=self.restartGame)

    # ─────────────────────────────────────────
    #  Rendering
    # ─────────────────────────────────────────
    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def render(self):
        if RENDER_MODE:
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


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()
