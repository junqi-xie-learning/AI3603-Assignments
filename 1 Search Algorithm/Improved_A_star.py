import DR20API
import numpy as np
import heapq

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.
def next_pos(current_map, current_pos):
    """
    Find all possible next positions of the current position.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.

    Return:
    candidates -- A list of candidates of possible next positions
    """
    candidates = []
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]])
    for direction in directions:
        candidate = tuple(np.array(current_pos) + direction)
        if current_map[candidate] == 0:
            candidates.append(candidate)
    return candidates

def distance(current_pos, goal_pos):
    """
    Calculate the distance between current position and goal position.
    Also used as the heuristic function for A* search.

    Arguments:
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    distance -- An integer representing the distance between current position and goal position.
    """
    return np.linalg.norm(np.array(current_pos) - np.array(goal_pos))

def cost(current_map, current_pos, candidate, prev_pos):
    """
    Calculate the cost to move from current position to goal position

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    candidate -- A 2D vector indicating the candidate of the next position of the robot.
    prev_pos -- A 2D vector indicating the previous position of the robot, None if the robot is in its initial position.

    Return:
    cost -- An integer representing the cost to move from current position to goal position with the following rules:
    * Base cost: physical distance between the 2 positions
    * Additional cost for moving beside an obstacle: 1 for each neighboring obstacles
    * Additional cost for steering: 1 for each $\pi / 4$ turned
      Note that the default direction for the robot is (1, 0)
    """
    current_ori = controller.get_robot_ori()
    prev_path = np.array(current_pos) - np.array(prev_pos) if prev_pos else np.array([np.sin(current_ori), np.cos(current_ori)])
    next_path = np.array(candidate) - np.array(current_pos)

    distance = np.linalg.norm(next_path)
    obstacles = (8 - len(next_pos(current_map, candidate))) / 4
    steering = 1 - prev_path.dot(next_path) / (np.linalg.norm(prev_path) * np.linalg.norm(next_path))

    distance_factor, obstacles_factor, steering_factor = 1, 1, 1
    return distance * distance_factor + obstacles * obstacles_factor + steering * steering_factor
###  END CODE HERE  ###

def Improved_A_star(current_map, current_pos, goal_pos):
    """
    Given current map of the world, current position of the robot and the position of the goal, 
    plan a path from current position to the goal using improved A* algorithm.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by improved A* algorithm.
    """

    ### START CODE HERE ###
    current_pos = tuple(current_pos)
    goal_pos = tuple(goal_pos)

    fringe, closed = [], {}
    backward, prev_pos = 0, None
    while current_pos != goal_pos:
        if current_pos not in closed:
            closed[current_pos] = prev_pos
            candidates = next_pos(current_map, current_pos)
            for candidate in candidates:
                weight = cost(current_map, current_pos, candidate, prev_pos)
                priority = backward + weight + distance(candidate, goal_pos)
                heapq.heappush(fringe, (priority, candidate, current_pos, backward + weight))
        _, current_pos, prev_pos, backward = heapq.heappop(fringe)
    closed[current_pos] = prev_pos
    
    path = []
    while current_pos:
        path.insert(0, current_pos)
        current_pos = closed[current_pos]
    ###  END CODE HERE  ###
    return path

def reach_goal(current_pos, goal_pos):
    """
    Given current position of the robot, 
    check whether the robot has reached the goal.

    Arguments:
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    is_reached -- A bool variable indicating whether the robot has reached the goal, where True indicating reached.
    """

    ### START CODE HERE ###
    is_reached = distance(current_pos, goal_pos) <= 1
    ###  END CODE HERE  ###
    return is_reached

if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.
    goal_pos = [100, 100]
    controller = DR20API.Controller()

    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    current_map = controller.update_map()

    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal.
        path = Improved_A_star(current_map, current_pos, goal_pos)
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

    # Stop the simulation.
    controller.stop_simulation()