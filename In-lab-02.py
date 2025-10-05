from collections import deque
import random

# A class to represent a single configuration of the 8-puzzle.
class PuzzleState:
    def __init__(self, layout, previous_state=None):
       
        self.layout = layout
        self.previous_state = previous_state

def generate_child_states(current_puzzle_state):

    child_states = []
    empty_pos = current_puzzle_state.layout.index(0)
    
    # Determine possible moves by swapping the empty tile.
    possible_swaps = []
    if empty_pos % 3 > 0: 
        possible_swaps.append(-1)
    if empty_pos % 3 < 2: 
        possible_swaps.append(1)
    if empty_pos // 3 > 0: 
        possible_swaps.append(-3)
    if empty_pos // 3 < 2: 
        possible_swaps.append(3)

    # Create new PuzzleState objects for each valid move.
    for move in possible_swaps:
        swap_pos = empty_pos + move
        child_layout = list(current_puzzle_state.layout)
        child_layout[empty_pos], child_layout[swap_pos] = child_layout[swap_pos], child_layout[empty_pos]
        child_states.append(PuzzleState(child_layout, current_puzzle_state))

    return child_states

def check_solvability(board_layout):

    inversion_count = 0
    tiles_only = [tile for tile in board_layout if tile != 0]
    
    for i in range(len(tiles_only)):
        for j in range(i + 1, len(tiles_only)):
            if tiles_only[i] > tiles_only[j]:
                inversion_count += 1
    
    return inversion_count % 2 == 0

def solve_puzzle_bfs(initial_layout, target_layout):

    root_state = PuzzleState(initial_layout)
    # The frontier holds states to be explored.
    frontier = deque([root_state])
    explored_configs = set()
    explored_count = 0

    while frontier:
        current_state = frontier.popleft()
        
        current_tuple = tuple(current_state.layout)
        if current_tuple in explored_configs:
            continue
        explored_configs.add(current_tuple)
        explored_count += 1

        # Check if the goal has been reached.
        if current_state.layout == target_layout:
            solution_steps = []
            # Backtrack from the goal to the start to build the path.
            while current_state:
                solution_steps.append(current_state.layout)
                current_state = current_state.previous_state
            
            print(f"States explored to find solution: {explored_count}")
            # The path is built backward, so it needs to be reversed.
            return solution_steps[::-1]

        for child in generate_child_states(current_state):
            if tuple(child.layout) not in explored_configs:
                frontier.append(child)

    return None 

def create_scrambled_puzzle(solved_layout, depth):

    current_node = PuzzleState(solved_layout)
    current_depth = 0
    while current_depth < depth:
        next_possible_states = generate_child_states(current_node)
        scrambled_state = random.choice(next_possible_states).layout
        if check_solvability(scrambled_state):
            current_node = PuzzleState(scrambled_state)
            current_depth += 1
            
    return current_node.layout

if __name__ == "__main__":
    SOLVED_PUZZLE = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    SCRAMBLE_DEPTH = 20

    target_puzzle = create_scrambled_puzzle(SOLVED_PUZZLE, SCRAMBLE_DEPTH)

    print("Initial (Solved) State:", SOLVED_PUZZLE)
    print(f"Target Puzzle (Scrambled to depth {SCRAMBLE_DEPTH}):", target_puzzle)
    print("-" * 30)

    # Find the solution using BFS.
    solution_path = solve_puzzle_bfs(SOLVED_PUZZLE, target_puzzle)

    if solution_path:
        print("\nSolution Path Found:")
        for i, step in enumerate(solution_path):
            print(f"Step {i}: {step}")
    else:
        print("No solution could be found.")