from constants import UP, DOWN, LEFT, RIGHT
import random

class AgentP: #class to handle the agent's current state, It observes the game state 
    def __init__(self):
        self.weights = {} #Q-table: (state, action) -> Qvalue
        self.alpha = 0.1 #learning rate
        self.gamma = 0.9 #discount factor
        self.epsilon = 0.1 #exploration rate
        self.last_state = None
        self.last_action = None

    def get_state(self, pacman, ghosts):
        self.current_ghosts = ghosts

        node = pacman.node #It gets the current node where Pac-Man is located

        node_x=int(node.position.x) #extracts the x and y coord of the pacmans current position, converted to int so the state is discrete
        node_y=int(node.position.y)

        up_open = 1 if node.neighbors.get(UP) is not None else 0 # Checks if there is a free connected node in the UP direction from the current node (1 if yes, 0 if no)
        down_open = 1 if node.neighbors.get(DOWN) is not None else 0 #
        left_open = 1 if node.neighbors.get(LEFT) is not None else 0 #
        right_open = 1 if node.neighbors.get(RIGHT) is not None else 0 #
        
        #ghost code
        #ghost_near = 0
        #for ghost in ghosts:
         #   dx = abs(pacman.node.position.x - ghost.node.position.x)
          #  dy = abs(pacman.node.position.y - ghost.node.position.y)

           # if dx + dy < 40: #can be changed
            #    ghost_near = 1 # if ghost is close then its true else 0
             #   break
        #min_distance = 9999

        #for ghost in ghosts:
        #    dx = abs(pacman.node.position.x - ghost.node.position.x)
        #    dy = abs(pacman.node.position.y - ghost.node.position.y)
        #    dist = dx + dy

        #    if dist < min_distance:
        #        min_distance = dist

        # normalisera så värdet inte blir för stort
        #ghost_distance = min(min_distance, 200) / 200.0
        
        #state = (node_x, node_y, up_open, down_open, left_open, right_open, ghost_near)

        #state = (node_x, node_y, up_open, down_open, left_open, right_open, ghost_distance)
        
        state = (node_x, node_y, up_open, down_open, left_open, right_open)

        return state
    
    def get_features(self, state, action):
        #node_x, node_y, up, down, left, right, ghost_distance = state
        node_x, node_y, up, down, left, right = state

        features = {}

        # Bias term
        features["bias"] = 1.0

        next_x = node_x
        next_y = node_y

        if action == UP:
            next_y -= 16
        elif action == DOWN:
            next_y += 16
        elif action == LEFT:
            next_x -= 16
        elif action == RIGHT:
            next_x += 16

        min_distance = 9999

        for ghost in self.current_ghosts:
            gx = int(ghost.node.position.x)
            gy = int(ghost.node.position.y)

            dist = abs(next_x - gx) + abs(next_y - gy)

            if dist < min_distance:
                min_distance = dist

        # normalisera
        ghost_risk = min(min_distance, 200) / 200.0

        # Ghost feature
        #features["ghost_distance"] = ghost_distance

        # Direction features
        features["move_up"] = 1.0 if action == UP else 0.0
        features["move_down"] = 1.0 if action == DOWN else 0.0
        features["move_left"] = 1.0 if action == LEFT else 0.0
        features["move_right"] = 1.0 if action == RIGHT else 0.0
        features["ghost_risk"] = ghost_risk

        return features
    
    def get_q_value(self, state, action):
        features = self.get_features(state, action)
        q = 0.0

        for f in features:
            weight = self.weights.get(f, 0.0)
            q += weight * features[f]

        return q
    
    def get_action(self,state):
        _, _, up, down, left, right = state #ignore x y position_ _

        actions = []
        if up == 1:    #if there is way up then put UP on the list
            actions.append(UP)
        if down == 1:
            actions.append(DOWN)
        if left == 1:
            actions.append(LEFT)
        if right == 1:
            actions.append(RIGHT)

        if len(actions) == 0:
            return None #not valid move, no way to move

        ####
        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            action = max(actions, key=lambda a: self.get_q_value(state, a))    
        
        self.last_state = state
        self.last_action = action
        return action
    

    def get_reward(self, pacman, pellets_eaten, dead):
        if dead:
            return -100
        if pellets_eaten:
            return 10
        return -1
        
    
    
    def update_q(self, reward, new_state):
        if self.last_state is None or self.last_action is None:
            return
        
        # möjliga actions i new_state
        _, _, up, down, left, right = new_state
        possible_actions = []
        if up == 1: possible_actions.append(UP)
        if down == 1: possible_actions.append(DOWN)
        if left == 1: possible_actions.append(LEFT)
        if right == 1: possible_actions.append(RIGHT)

        if len(possible_actions) == 0:
            max_next_q = 0.0
        else:
            max_next_q = max(self.get_q_value(new_state, a) for a in possible_actions)

        current_q = self.get_q_value(self.last_state, self.last_action)

        difference = reward + self.gamma * max_next_q - current_q

        features = self.get_features(self.last_state, self.last_action)

        for f in features:
            old_weight = self.weights.get(f, 0.0)
            self.weights[f] = old_weight + self.alpha * difference * features[f]