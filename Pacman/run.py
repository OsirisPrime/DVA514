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
from sprites import LifeSprites
from sprites import MazeSprites
import matplotlib.pyplot as plt
from pacman import episode_pellets, episode_time
from agentPac import AgentP


class GameController(object):
    def __init__(self): 
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(False)
        self.level = 0
        self.lives = NUMLIVES
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.nextDirection = STOP
        self.lastPelletCount = 0

        self.agent = AgentP()#new
        self.episode_count = 0
        self.max_episodes = 2000
        self.ghost_eaten_flag = False
        self.fruit_eaten_flag = False

    def restartGame(self):
        self.lives = NUMLIVES
        self.level = 0
        self.pause.paused = False
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        #self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)

    def resetLevel(self):
        self.pause.paused = False
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        #self.textgroup.showText(READYTXT)

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = False
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)

    def startGame(self):
        self.setBackground()
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.background = self.mazesprites.constructBackground(self.background, self.level%5)
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0,17), (27,17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)

    def update(self):
        dt = self.clock.tick(FPS) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:

            if self.pacman.overshotTarget(): #new, pacman is in a node and can change direction
                old_state = self.agent.get_state(self.pacman, self.ghosts, self.pellets, self.fruit) #agent reads current state/position
                action = self.agent.get_action(old_state) #agent choose direction based on current state
                
                if action is not None:
                    self.pacman.nextDirection = action #new saves the agenst choosment.

                self.pacman.update(dt)
                self.ghosts.update(dt)
                if self.fruit is not None:
                    self.fruit.update(dt)
                
                #calculate new_state and reward
                
                new_state = self.agent.get_state(self.pacman, self.ghosts, self.pellets,self.fruit)
                
                pellets_eaten=self.pellets.numEaten > self.lastPelletCount#new, checks if pacman ate a pellet(true,false)
                self.lastPelletCount = self.pellets.numEaten #saves amoint of eaten pelets, checks next frame if the amount pellets have increased
            
                dead = not self.pacman.alive #checks if pacman has died (trie or false)
                
                reward = self.agent.get_reward(self.pacman, pellets_eaten, dead, self.ghost_eaten_flag, self.fruit_eaten_flag)#new sends pacmans state, amount of eaten pellets, if pacman have died or not, den its gets reward or not. 
                self.ghost_eaten_flag = False
                self.fruit_eaten_flag = False
                self.agent.update_q(reward, new_state)
                print("Old state:", self.agent.last_state)
                print("New state:", new_state)
                print("Reward:", reward)
                #print("Q-table size:", len(self.agent.weights))
                print("-----------")
            else:
                # Om inte i node → bara fortsätt rörelsen
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
        

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                self.plotResults()
                self.agent.save_weights()
                quit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            self.hideEntities()

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.ghost_eaten_flag = True
                    self.pacman.visible = False
                    self.updateScore(ghost.points)
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    ghost.visible = False
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                     if self.pacman.alive:
                         episode_pellets.append(self.pacman.pellets_eaten)
                         episode_time.append(self.pacman.time_alive)
                         self.pacman.pellets_eaten = 0
                         self.pacman.time_alive = 0.0

                         self.lives -=  1
                         self.lifesprites.removeImage()
                         self.pacman.die()
                         self.episode_count += 1
                         #self.agent.epsilon = max(0.05, self.agent.epsilon * 0.995) #en av de sista sakerna som man har gjort decay epsilon
                         print("Episode:", self.episode_count)
                         if self.episode_count >= self.max_episodes:
                            print("Training finished")
                            self.plotResults()
                            self.agent.save_weights()
                            pygame.quit()
                            quit()

                         self.ghosts.hide()

                         self.restartGame()
                         #if self.lives <= 0:
                             #self.textgroup.showText(GAMEOVERTXT)
                             #self.pause.setPause(pauseTime=3, func=self.restartGame)
                         #    self.restartGame()
                         #else:
                             #self.pause.setPause(pauseTime=3, func=self.resetLevel)
                         #    self.resetLevel()

    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20))
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                self.fruit_eaten_flag = True
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

                self.episode_count += 1
                #self.agent.epsilon = max(0.1, self.agent.epsilon * 0.999)#epsilon decay 
                print("Episode:", self.episode_count)

                if self.episode_count >= self.max_episodes:
                    print("Training finished")
                    self.plotResults()
                    self.agent.save_weights()
                    pygame.quit()
                    quit()

                self.nextLevel()

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def render(self):
        self.screen.blit(self.background, (0, 0))
        # self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))
        pygame.display.update()

    #def plotResults(self): #new
    #    plt.figure()
    #    plt.plot(episode_pellets, marker='o')
    #    plt.xlabel("Episode (1 life = 1 episode)")
    #    plt.ylabel("Pellets Collected")
    #    plt.title("Trained agent (Q-learning)")
    #    plt.show()

    #    plt.figure()
    #    plt.plot(episode_time, marker='o')
    #    plt.xlabel("Episode 1 life = 1 episode")
    #    plt.ylabel("Time alive (s)")
    #    plt.title("Trained agent (Q-learning)")
    #    plt.show()
    def plotResults(self):

        def moving_average(data, window_size=30):
            if len(data) < window_size:
                return []
            return [sum(data[i:i+window_size]) / window_size
                    for i in range(len(data) - window_size + 1)]

        # --------- PELLETS ---------
        plt.figure(figsize=(10,5))

        plt.plot(episode_pellets, alpha=0.3, label="Raw")

        ma_pellets = moving_average(episode_pellets, 30)
        if ma_pellets:
            plt.plot(range(29, len(episode_pellets)),
                    ma_pellets,
                    linewidth=2,
                    label="Moving Average (30)")

        plt.xlabel("Episode")
        plt.ylabel("Pellets Collected")
        plt.title("Training Progress - Pellets")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


        # --------- TIME ALIVE ---------
        plt.figure(figsize=(10,5))

        plt.plot(episode_time, alpha=0.3, label="Raw")

        ma_time = moving_average(episode_time, 30)
        if ma_time:
            plt.plot(range(29, len(episode_time)),
                    ma_time,
                    linewidth=2,
                    label="Moving Average (30)")

        plt.xlabel("Episode")
        plt.ylabel("Time Alive (s)")
        plt.title("Training Progress - Time Alive")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()
