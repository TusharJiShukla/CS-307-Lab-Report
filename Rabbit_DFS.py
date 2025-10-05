from collections import deque

def is_valid(state):
    empty = state.index(-1)
    if state == (0, 0, 0, -1, 1, 1, 1):
        return True
    if empty == 0 and state[1] == 1 and state[2] == 1:
        return False
    if empty == 6 and state[5] == 0 and state[4] == 0:
        return False
    if empty == 1 and state[0] == 0 and state[2] == 1 and state[3] == 1:
        return False
    if empty == 5 and state[6] == 1 and state[4] == 0 and state[3] == 0:
        return False
    if state[empty-1] == 0 and state[empty-2] == 0 and state[empty+1] == 1 and state[empty+2] == 1:
        return False
    return True

def swap(state, i, j):
    new_state = list(state)
    temp = new_state[i]
    new_state[i] = new_state[j]
    new_state[j] = temp
    return tuple(new_state)

def get_successors(state):
    successors = []
    empty = state.index(-1)
    moves = [-2, -1, 1, 2]
    for move in moves:
        if 0 <= empty + move < 7:
            new_state = swap(state, empty, empty + move)
            mid = empty + move // 2 if abs(move) == 2 else None
            if mid is not None and state[mid] == -1:
                continue  # Skip if jumping over empty (though rare)
            if move > 0 and state[empty + move] == 0:  # O (0) moving left
                if is_valid(new_state):
                    successors.append(new_state)
            elif move < 0 and state[empty + move] == 1:  # E (1) moving right
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
        state = frontier.pop()  # LIFO for DFS
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

start_state = (1, 1, 1, -1, 0, 0, 0)
goal_state = (0, 0, 0, -1, 1, 1, 1)

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
Total Number Of Nodes Visited: 30
Max Size Of queue at a point was: 6
Solution found:
Number Of nodes in solution: 16
(1, 1, 1, -1, 0, 0, 0)
(1, 1, 1, 0, -1, 0, 0)
(1, 1, -1, 0, 1, 0, 0)
(1, -1, 1, 0, 1, 0, 0)
(1, 0, 1, -1, 1, 0, 0)
(1, 0, 1, 0, 1, -1, 0)
(1, 0, 1, 0, 1, 0, -1)
(1, 0, 1, 0, -1, 0, 1)
(1, 0, -1, 0, 1, 0, 1)
(-1, 0, 1, 0, 1, 0, 1)
(0, -1, 1, 0, 1, 0, 1)
(0, 0, 1, -1, 1, 0, 1)
(0, 0, 1, 0, 1, -1, 1)
(0, 0, 1, 0, -1, 1, 1)
(0, 0, -1, 0, 1, 1, 1)
(0, 0, 0, -1, 1, 1, 1)
'''