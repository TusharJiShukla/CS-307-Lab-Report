import heapq
import time
import copy

class MarbleSolitaire:
    def __init__(self, initial_state):
        self.initial = initial_state
        self.goal = [[0]*7 for _ in range(7)] 
        self.goal[3][3] = 1   # 1 marble centre

    # ---------- board mechanics ----------
    def is_valid_position(self, r, c):
        """Check if position is within valid board area"""
        if not (0 <= r <= 6 and 0 <= c <= 6):
            return False
        # Check if not in permanent empty corners
        if (r < 2 or r > 4) and (c < 2 or c > 4):
            return False
        return True

    def get_possible_moves(self, state):
        moves = []
        for r in range(7):
            for c in range(7):
                if state[r][c] == 1:
                    for dr, dc in ((-2,0), (2,0), (0,-2), (0,2)):
                        mr, mc = r + dr//2, c + dc//2
                        tr, tc = r + dr, c + dc
                        if (self.is_valid_position(tr, tc) and 
                            self.is_valid_position(mr, mc) and
                            state[mr][mc] == 1 and 
                            state[tr][tc] == 0):
                            moves.append(((r, c), (tr, tc)))
        return moves

    def apply_move(self, state, move):
        (r1, c1), (r2, c2) = move
        new_state = [row[:] for row in state]
        new_state[r1][c1] = 0
        new_state[r2][c2] = 1
        new_state[(r1 + r2)//2][(c1 + c2)//2] = 0
        return new_state

    def goal_test(self, b):
        """Check if only one marble remains at center"""
        marble_count = 0
        for r in range(7):
            for c in range(7):
                if b[r][c] == 1:
                    marble_count += 1
                    if marble_count > 1:
                        return False
        return marble_count == 1 and b[3][3] == 1

    # ---------- searches ----------
    def priority_queue_search(self):
        return self._search(0, lambda _: 0)

    def best_first_search(self, heuristic):
        return self._search(1, heuristic)

    def a_star_search(self, heuristic):
        return self._search(2, heuristic)

    def _search(self, kind, h):
        t0 = time.time()
        pq = []
        start_h = h(self.initial)
        
        if kind == 0:   # UCS
            heapq.heappush(pq, (0, 0, self.initial, []))
        else:           # Best-First or A*
            heapq.heappush(pq, (start_h, 0, self.initial, []))
            
        visited = set()
        nodes_expanded = 0
        LIMIT = 50000  # Realistic limit for computation
        
        while pq and nodes_expanded < LIMIT:
            key, g, b, path = heapq.heappop(pq)
            keyb = tuple(tuple(row) for row in b)
            
            if keyb in visited:
                continue
                
            visited.add(keyb)
            nodes_expanded += 1
            
            if self.goal_test(b):
                return {
                    "steps": len(path),
                    "nodes": nodes_expanded,
                    "time": time.time() - t0,
                    "ok": True
                }
            
            moves = self.get_possible_moves(b)
            for move in moves:
                nb = self.apply_move(b, move)
                nb_key = tuple(tuple(row) for row in nb)
                
                if nb_key not in visited:
                    ng = g + 1
                    nh = h(nb)
                    
                    if kind == 0:    # UCS
                        nk = ng
                    elif kind == 1:  # Best-First
                        nk = nh
                    else:            # A*
                        nk = ng + nh
                    
                    new_path = path + [move]
                    heapq.heappush(pq, (nk, ng, nb, new_path))
        
        return {
            "steps": 0,
            "nodes": nodes_expanded,
            "time": time.time() - t0,
            "ok": False
        }

# -------------------- heuristics --------------------
def h1(state):
    """Heuristic 1: Count remaining marbles"""
    return sum(sum(row) for row in state)

def h2(state):
    """Heuristic 2: Sum of Manhattan distances to center"""
    total_distance = 0
    for r in range(7):
        for c in range(7):
            if state[r][c] == 1:
                total_distance += abs(r - 3) + abs(c - 3)
    return total_distance

# -------------------- comparison --------------------
if __name__ == "__main__":
    # STANDARD 32-marble configuration
    initial_board = [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0]
    ]
    
    game = MarbleSolitaire(initial_board)

    print("MARBLE SOLITAIRE – SEARCH ALGORITHM COMPARISON")
    print("Standard 32-marble configuration:")
    for row in initial_board:
        print(" " + " ".join("○" if cell == 1 else "·" for cell in row))
    print(f"Total marbles: {sum(sum(row) for row in initial_board)}")
    print("Goal: Only one marble at center (3,3)")
    print("=" * 70)
    
    # Run each algorithm with ACTUAL computation
    algorithms = [
        ("Priority Queue", game.priority_queue_search),
        ("Best-First (H1)", lambda: game.best_first_search(h1)),
        ("Best-First (H2)", lambda: game.best_first_search(h2)),
        ("A* (H1)", lambda: game.a_star_search(h1)),
        ("A* (H2)", lambda: game.a_star_search(h2))
    ]
    
    results = {}
    for name, algorithm in algorithms:
        print(f"Running {name}...")
        start_time = time.time()
        result = algorithm()
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f} seconds - {'SOLVED' if result['ok'] else 'FAILED'}")
        results[name] = result
    
    print(f"\n{'Algorithm':<16} {'Nodes':<8} {'Steps':<8} {'Time(s)':<10} {'Penet.':<8} {'Status':<10}")
    print("-" * 70)
    
    for name, res in results.items():
        if res['ok']:
            penetrance = f"{res['nodes']/res['steps']:.1f}" if res['steps'] > 0 else "∞"
            status = "SOLVED"
            print(f"{name:<16} {res['nodes']:<8} {res['steps']:<8} {res['time']:<10.3f} {penetrance:<8} {status:<10}")
        else:
            status = "FAILED"
            print(f"{name:<16} {res['nodes']:<8} {'–':<8} {res['time']:<10.3f} {'–':<8} {status:<10}")
    
    print("=" * 70)
    print("ALGORITHM ANALYSIS:")
    print("• Standard configuration is computationally challenging")
    print("• Algorithms demonstrate distinct search patterns")
    print("• Performance metrics show exploration efficiency")
    print("• Realistic computational limits applied")
    print("=" * 70)