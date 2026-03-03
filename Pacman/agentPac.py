from constants import UP, DOWN, LEFT, RIGHT, FREIGHT
import random
import csv

class AgentP: #class to handle the agent's current state, It observes the game state 
    def __init__(self, alpha=0.2, epsilon=0.2, gamma=0.9):
        self.weights = {} #Q-table: (state, action) -> Qvalue
        self.alpha = float(alpha) #learning rate
        self.gamma = float(gamma) #discount factor
        self.epsilon = float(epsilon)#exploration rate
        self.last_state = None
        self.last_action = None

    def get_state(self, pacman, ghosts, pellets,fruit):
        self.current_ghosts = ghosts
        self.current_pellets = pellets
        self.current_fruit = fruit

        #print("Number of pellets:", len(self.current_pellets.pelletList))

        node = pacman.node #It gets the current node where Pac-Man is located

        node_x=int(node.position.x) #extracts the x and y coord of the pacmans current position, converted to int so the state is discrete
        node_y=int(node.position.y)

        up_open = 1 if node.neighbors.get(UP) is not None else 0 # Checks if there is a free connected node in the UP direction from the current node (1 if yes, 0 if no)
        down_open = 1 if node.neighbors.get(DOWN) is not None else 0 #
        left_open = 1 if node.neighbors.get(LEFT) is not None else 0 #
        right_open = 1 if node.neighbors.get(RIGHT) is not None else 0 #

        #print("Computed:", up_open, down_open, left_open, right_open)
        #print("------")
        
        state = (node_x, node_y, up_open, down_open, left_open, right_open)

        return state
    
    def get_features(self, state, action):
        #node_x, node_y, up, down, left, right, ghost_distance = state
        node_x, node_y, up, down, left, right = state

        features = {} #a feature vector for the dif features. 

        # Bias term
        features["Bias"] = 1.0 #constant feature, 1.0 så att vikten kan påverka q direkt. 

        next_x = node_x
        next_y = node_y

        if action == UP:
            next_y -= 16
        elif action == DOWN:
            next_y += 16
        elif action == LEFT:
            next_x -= 16
        elif action == RIGHT:
            next_x += 16      #16 på grund av tilewidth, +16 right enligt koordinat systemet + höger - vänster, matcha cordinater

        min_distance = float("inf")
        ghost_close = 0.0
        ghost_in_front = 0.0
        scared_feature = 0.0

        for ghost in self.current_ghosts:
            gx = int(ghost.node.position.x)
            gy = int(ghost.node.position.y)

            #print("Ghost:", ghost, "Position:", gx, gy)

            dist = abs(next_x - gx) + abs(next_y - gy)

            #print("Pacman next position:", next_x, next_y)
            #print("Ghost position:", gx, gy)
            #print("Distance:", dist)
            #print("-----")

            if dist < min_distance:
                min_distance = dist
            
            
            # Ghost one step away (danger only if not scared)
            if ghost.mode.current != FREIGHT:
                if dist <= 16:
                    ghost_close = -1.0
            
            # Ghost in chosen direction
            if ghost.mode.current != FREIGHT:
                if action == LEFT and gy == node_y and gx < node_x:
                    ghost_in_front = -1.0
                elif action == RIGHT and gy == node_y and gx > node_x:
                    ghost_in_front = -1.0
                elif action == UP and gx == node_x and gy < node_y:
                    ghost_in_front = -1.0
                elif action == DOWN and gx == node_x and gy > node_y:
                    ghost_in_front = -1.0
            
            if ghost.mode.current == FREIGHT: #if scared ghost attack
                scared_feature = 1.0
                
                
        # normalisera
        normalized = min(min_distance, 200) / 200.0
        ghost_risk = normalized * 2.0 - 1.0 #skalar om intervallet till [-1,1]

       ############################################################################### 

        min_food_distance = float("inf")

        for pellet in self.current_pellets.pelletList:
            px = int(pellet.position.x)
            py = int(pellet.position.y)

            dista = abs(next_x - px) + abs(next_y - py)

            if dista < min_food_distance:
                min_food_distance = dista

        if min_food_distance == float("inf"):
            food_feature = 0.0
        else:
            normalized_food = min(min_food_distance, 200) / 200.0
            food_feature = 1.0 - normalized_food  # nära mat = 1
            food_feature = food_feature * 2.0 - 1.0
        
        
        fruit_feature = 0.0

        if self.current_fruit is not None:
            fx = int(self.current_fruit.position.x)
            fy = int(self.current_fruit.position.y)

            dist = abs(next_x - fx) + abs(next_y - fy)
            normalized_fruit = min(dist, 200) / 200.0
            fruit_feature = 1.0 - normalized_fruit
            fruit_feature = fruit_feature * 2.0 - 1.0

        features["Food_Closeness"] = food_feature
        features["Fruit_Closeness"] = fruit_feature
        features["Move_Up"] = 1.0 if action == UP else 0.0
        features["Move_Down"] = 1.0 if action == DOWN else 0.0
        features["Move_Left"] = 1.0 if action == LEFT else 0.0
        features["Move_Right"] = 1.0 if action == RIGHT else 0.0
        features["Ghost_Risk"] = ghost_risk
        features["Ghost_One_Step"] = ghost_close
        features["Ghost_In_Front"] = ghost_in_front
        features["Scared_Ghost"] = scared_feature

        return features
    
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
    
    def get_q_value(self, state, action):
        features = self.get_features(state, action)
        q = 0.0

        for feature in features:
            weight = self.weights.get(feature, 0.0)
            q += weight * features[feature]

        return q
    

    def get_reward(self, pacman, pellets_eaten, dead, ghost_eaten, fruit_eaten): #new updates
        if dead:
            return -150
        if ghost_eaten:
            return 100
        if fruit_eaten:
            return 30
        if pellets_eaten:
            return 20
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

        if len(possible_actions) == 0: ##maxQ(s',a') max nästa värde
            max_next_q = 0.0
        else:
            max_next_q = max(self.get_q_value(new_state, a) for a in possible_actions)

        current_q = self.get_q_value(self.last_state, self.last_action)#calculates the state befor

        difference = reward + self.gamma * max_next_q - current_q #self gamma=self.discount

        features = self.get_features(self.last_state, self.last_action)

        for f in features:
            old_weight = self.weights.get(f, 0.0)
            self.weights[f] = old_weight + self.alpha * difference * features[f] #updates the wieghts

    
    def save_weights(self, filename = "Wieghts.csv"):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Feature", "Weight"])
            for key, value in self.weights.items():
                writer.writerow([key,value])