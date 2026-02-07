import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites
import random

episode_pellets = [0]
episode_time = [0.0]

class Pacman(Entity):
    def __init__(self, node):
        Entity.__init__(self, node)
        self.name = PACMAN
        self.directions = {STOP: Vector2(), UP: Vector2(0, -1), DOWN: Vector2(0, 1), LEFT: Vector2(-1, 0),
                           RIGHT: Vector2(1, 0)}
        self.direction = STOP
        self.speed = 100 * TILEWIDTH / 16
        self.radius = 10
        self.color = YELLOW
        self.direction = LEFT
        self.nextDirection = STOP #new
        self.time_alive = 0.0 #new
        self.setBetweenNodes(LEFT)
        self.node = node
        # self.setPosition()
        self.target = node
        self.collideRadius = 5
        self.alive = True
        self.sprites = PacmanSprites(self)
        
        self.steps = 0 #new
        self.pellets_eaten = 0 #new

    def setPosition(self):
        self.position = self.node.position.copy()

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True

    def die(self):
        #episode_pellets.append(self.pellets_eaten)
        #episode_time.append(self.time_alive)
        
        self.alive = False
        self.direction = STOP
        #self.steps = 0
        #self.pellets_eaten = 0
        #self.time_alive = 0.0

    def update(self, dt):

        self.steps +=1
        self.time_alive += dt
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt

        if self.overshotTarget():
            self.node = self.target

            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]

            self.nextDirection = self.getValidKey() #if its close to a wall peak up a direction

            self.target = self.getNewTarget(self.nextDirection)
            if self.target is not self.node:
                self.direction = self.nextDirection
            else:
                self.target = self.getNewTarget(self.direction)
            
            if self.target is self.node:
                self.direction = STOP
            
            self.setPosition()

    #def update(self, dt):
    #    self.sprites.update(dt)
    #    self.position += self.directions[self.direction] * self.speed * dt
    #    direction = self.getValidKey()
    #    if self.overshotTarget():
    #        self.node = self.target
    #        if self.node.neighbors[PORTAL] is not None:
    #            self.node = self.node.neighbors[PORTAL]
    #        self.target = self.getNewTarget(direction)
    #        if self.target is not self.node:
    #            self.direction = direction
    #        else:
    #            self.target = self.getNewTarget(self.direction)
    #        if self.target is self.node:
    #            self.direction = STOP
    #        self.setPosition()
    #    else:
    #        if self.oppositeDirection(direction):
    #            self.reverseDirection()

    def validDirection(self, direction):
        if direction is not STOP:
            if self.node.neighbors[direction] is not None:
                return True
        return False

    def getNewTarget(self, direction):
        if self.validDirection(direction):
            return self.node.neighbors[direction]
        return self.node
    
    def getValidKey(self):
        return random.choice([UP,DOWN,LEFT,RIGHT])
    

    ##def getValidKey(self):
        #key_pressed = pygame.key.get_pressed()
        #if key_pressed[K_UP]:
        #    return UP
        #if key_pressed[K_DOWN]:
        #    return DOWN
        #if key_pressed[K_LEFT]:
        #    return LEFT
        #if key_pressed[K_RIGHT]:
        #    return RIGHT
        #return STOP

    def overshotTarget(self):
        if self.target is not None:
            vec1 = self.target.position - self.node.position
            vec2 = self.position - self.node.position
            node2Target = vec1.magnitudeSquared()
            node2Self = vec2.magnitudeSquared()
            return node2Self >= node2Target
        return False

    def reverseDirection(self):
        self.direction *= -1
        temp = self.node
        self.node = self.target
        self.target = temp

    def oppositeDirection(self, direction):
        if direction is not STOP:
            if direction == self.direction * -1:
                return True
        return False

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                self.pellets_eaten +=1 #new, updates how many pellets eaten
                return pellet
        return None

    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius) ** 2
        if dSquared <= rSquared:
            return True
        return False