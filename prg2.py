import heapq

def astar(grid, start, goal):

    rows = len(grid)
    cols = len(grid[0])

    def heuristic(a, b):
        # Manhattan Distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_cost = {start: 0}

    while open_list:

        current = heapq.heappop(open_list)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        # Move in 4 directions
        neighbors = [(0,1),(1,0),(0,-1),(-1,0)]

        for dx, dy in neighbors:
            next_node = (current[0] + dx, current[1] + dy)

            if 0 <= next_node[0] < rows and 0 <= next_node[1] < cols:
                
                # Skip obstacles
                if grid[next_node[0]][next_node[1]] == 1:
                    continue

                new_cost = g_cost[current] + 1

                if next_node not in g_cost or new_cost < g_cost[next_node]:
                    g_cost[next_node] = new_cost
                    f_cost = new_cost + heuristic(next_node, goal)
                    heapq.heappush(open_list, (f_cost, next_node))
                    came_from[next_node] = current

    return "No Path Found"


# 0 = free path, 1 = obstacle
grid = [
    [0,0,0,0,0],
    [1,1,0,1,0],
    [0,0,0,1,0],
    [0,1,0,0,0],
    [0,0,0,1,0]
]

start = (0,0)
goal = (4,4)

path = astar(grid, start, goal)

print("The Path is:")
for p in path:
    print("->", p, end=" ")