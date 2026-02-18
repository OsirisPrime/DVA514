import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites


class Pacman(Entity):
    def __init__(self, node):
        Entity.__init__(self, node)
        self.name = PACMAN
        self.directions = {STOP: Vector2(), UP: Vector2(0, -1), DOWN: Vector2(0, 1), LEFT: Vector2(-1, 0), RIGHT: Vector2(1, 0)}
        self.direction = STOP
        self.speed = 100 * TILEWIDTH / 16
        self.radius = 10
        self.color = YELLOW
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.node = node
        self.target = node
        self.collideRadius = 5
        self.alive = True
        self.sprites = PacmanSprites(self)
        # AI stores its *requested* direction here; the node system applies
        # it only when Pac-Man reaches the next tile intersection, exactly
        # as a keyboard press would be — prevents wall clipping.
        self.ai_direction = LEFT

    def setPosition(self):
        self.position = self.node.position.copy()

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.ai_direction = LEFT

    def die(self):
        self.alive = False
        self.direction = STOP

    def update(self, dt):
        self.sprites.update(dt)
        self.position += self.directions[self.direction] * self.speed * dt
        # ai_direction is the *requested* direction from the agent.
        # Treating it the same as a keyboard press means it only takes effect
        # at tile intersections (overshotTarget), so the node-graph always
        # validates the move — identical to how human input works.
        direction = self.ai_direction if USE_AI else self.getValidKey()
        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)
            if self.target is self.node:
                self.direction = STOP
            self.setPosition()
        else:
            if self.oppositeDirection(direction):
                self.reverseDirection()

    def validDirection(self, direction):
        if direction is not STOP:
            # Must check node access control, not just neighbour existence.
            # Entity.validDirection does the same — Pac-Man's original override
            # omitted this, which let the AI (and keyboard play) walk straight
            # into the ghost home area despite denyHomeAccess() being called.
            if self.name in self.node.access[direction]:
                if self.node.neighbors[direction] is not None:
                    return True
        return False

    def getNewTarget(self, direction):
        if self.validDirection(direction):
            return self.node.neighbors[direction]
        return self.node

    def getValidKey(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return STOP

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