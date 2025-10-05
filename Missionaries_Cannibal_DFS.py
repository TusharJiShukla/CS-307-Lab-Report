from collections import deque

def is_valid(state):
    m_l, c_l, b = state
    m_r = 3 - m_l
    c_r = 3 - c_l
    if m_l < 0 or c_l < 0 or m_r < 0 or c_r < 0:
        return False
    if (m_l > 0 and m_l < c_l) or (m_r > 0 and m_r < c_r):
        return False
    return True

def get_successors(state):
    successors = []
    m_l, c_l, b = state
    dir = -1 if b == 1 else 1
    possible_moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
    for dm, dc in possible_moves:
        new_m_l = m_l + dir * dm
        new_c_l = c_l + dir * dc
        new_b = 1 - b
        new_state = (new_m_l, new_c_l, new_b)
        if is_valid(new_state):
            successors.append(new_state)
    return successors

def dfs(start_state, goal_state):
    frontier = deque([start_state])
    visited = set([start_state])
    parent = {start_state: None}
    expanded = 0
    max_size = 1
    while frontier:
        max_size = max(max_size, len(frontier))
        state = frontier.pop()
        expanded += 1
        if state == goal_state:
            path = []
            current = state
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            print(f"Total Number Of Nodes Visited: {expanded}")
            print(f"Max Size Of queue at a point was: {max_size}")
            return path
        for successor in get_successors(state):
            if successor not in visited:
                visited.add(successor)
                parent[successor] = state
                frontier.append(successor)
    return None

start_state = (3, 3, 1)
goal_state = (0, 0, 0)

solution = dfs(start_state, goal_state)
if solution:
    print("Solution found:")
    print(f"Number Of nodes in solution: {len(solution)}")
    for step in solution:
        print(step)
else:
    print("No solution found.")

'''
OUTPUT:
Total Number Of Nodes Visited: 12
Max Size Of queue at a point was: 4
Solution found:
Number Of nodes in solution: 12
(3, 3, 1)
(2, 2, 0)
(3, 2, 1)
(3, 0, 0)
(3, 1, 1)
(1, 1, 0)
(2, 2, 1)
(0, 2, 0)
(0, 3, 1)
(0, 1, 0)
(0, 2, 1)
(0, 0, 0)
'''