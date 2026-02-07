from constants import UP, DOWN, LEFT, RIGHT
class AgentP: #class to handle the agent's current state, It observes the game state 
    def __init__(self):
        pass

    def get_state(self, pacman):
        node = pacman.node #It gets the current node where Pac-Man is located

        node_x=int(node.position.x) #extracts the x and y coord of the pacmans current position, converted to int so the state is discrete
        node_y=int(node.position.y)

        up_open = 1 if node.neighbors.get(UP) is not None else 0 # Checks if there is a free connected node in the UP direction from the current node (1 if yes, 0 if no)

        down_open = 1 if node.neighbors.get(DOWN) is not None else 0 #
        left_open = 1 if node.neighbors.get(LEFT) is not None else 0 #
        right_open = 1 if node.neighbors.get(RIGHT) is not None else 0 #
        
        state = (node_x, node_y, up_open, down_open, left_open, right_open)

        return state