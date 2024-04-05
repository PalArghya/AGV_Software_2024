import heapq as q
class A_star:
    def __init__(self, parent, pos):
        self.pos = pos
        self.parent = parent
        self.g = 0
        self.ed = 0
        self.tot = 0
def euclidean(current, goal):
    dist = (((current[0]-goal[0])**2)+((current[1]-goal[1])**2))**0.5
    return dist
def astar(world, start, goal):
    evaluate = []
    visited = []
    start_node = A_star(start)
    goal_node = A_star(goal)
    q.heappush(evaluate, start_node) 
    while evaluate:
        current_node = q.heappop(evaluate)
        if current_node.position == goal_node.position:
            # If the current node is the goal node, construct and return the path
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[-1]  # Reverse the path to get it from start to goal
        evaluate.add(current_node.position)  # Add the current node to the closed set