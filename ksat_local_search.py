
# Hill-Climbing, Beam-Search (w=3,4), VND on uniform random 3-SAT

import random
import time

def generate_3sat(m, n):
    """Generate random 3-SAT instance"""
    clauses = []
    for _ in range(m):
        clause = []
        while len(clause) < 3:
            var = random.randint(1, n)
            lit = var if random.choice([True, False]) else -var
            if lit not in clause and -lit not in clause:
                clause.append(lit)
        clauses.append(clause)
    return clauses

def satisfied(clauses, assignment):
    """Count number of satisfied clauses"""
    count = 0
    for clause in clauses:
        for lit in clause:
            var = abs(lit) - 1
            if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                count += 1
                break
    return count

# Heuristic functions
def h1_sat(clauses, assignment):
    """H1: Negative satisfied clauses count (minimization)"""
    return -satisfied(clauses, assignment)

def h2_unsat(clauses, assignment):
    """H2: Unsatisfied clauses count (minimization)"""
    return len(clauses) - satisfied(clauses, assignment)

# Local search algorithms with improved strategies
def hill_climbing(clauses, n, heuristic, max_iter=1000):
    """Hill Climbing with random restarts"""
    start = time.time()
    best_success = 0
    best_steps = 0
    best_nodes = 0
    
    # Multiple random restarts
    for restart in range(10):
        current = [random.choice([True, False]) for _ in range(n)]
        current_val = heuristic(clauses, current)
        steps, nodes = 0, 0
        
        for iteration in range(max_iter // 10):
            steps += 1
            best_neighbor_val = current_val
            best_neighbor = None
            
            # Explore all 1-flip neighbors
            for i in range(n):
                nodes += 1
                neighbor = current.copy()
                neighbor[i] = not neighbor[i]
                neighbor_val = heuristic(clauses, neighbor)
                
                if neighbor_val < best_neighbor_val:
                    best_neighbor_val = neighbor_val
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                break  # Local optimum
                
            current, current_val = best_neighbor, best_neighbor_val
            
            # Check for solution
            if current_val == 0:
                best_success = 1
                best_steps = steps
                best_nodes = nodes
                break
                
        if best_success:
            break
    
    penetrance = best_nodes / best_steps if best_steps > 0 else 0
    elapsed = time.time() - start
    return best_success, best_steps, best_nodes, penetrance, elapsed

def beam_search(clauses, n, width, heuristic, max_iter=500):
    """Beam Search with beam width"""
    start = time.time()
    
    # Initialize beam with random states
    beam = [[random.choice([True, False]) for _ in range(n)] for _ in range(width)]
    steps, total_nodes = 0, 0
    success = 0
    
    for iteration in range(max_iter):
        steps += 1
        candidates = []
        
        # Generate all neighbors for all states in beam
        for state in beam:
            for i in range(n):
                total_nodes += 1
                neighbor = state.copy()
                neighbor[i] = not neighbor[i]
                neighbor_val = heuristic(clauses, neighbor)
                candidates.append((neighbor_val, neighbor))
        
        # Sort by heuristic value and select top 'width' candidates
        candidates.sort(key=lambda x: x[0])
        beam = [candidate[1] for candidate in candidates[:width]]
        
        # Check if any candidate is a solution
        if candidates[0][0] == 0:
            success = 1
            break
    
    penetrance = total_nodes / steps if steps > 0 else 0
    elapsed = time.time() - start
    return success, steps, total_nodes, penetrance, elapsed

def vnd(clauses, n, heuristic, max_iter=100):
    """Variable Neighborhood Descent with multiple neighborhoods"""
    start = time.time()
    
    current = [random.choice([True, False]) for _ in range(n)]
    current_val = heuristic(clauses, current)
    steps, total_nodes = 0, 0
    success = 0
    
    # neighborhood structures
    def neighborhood1(state):
        """1-flip neighborhood"""
        return [state[:i] + [not state[i]] + state[i+1:] for i in range(n)]
    
    def neighborhood2(state):
        """2-flip neighborhood"""
        neighbors = []
        for i in range(n):
            for j in range(i + 1, n):
                new_state = state.copy()
                new_state[i] = not new_state[i]
                new_state[j] = not new_state[j]
                neighbors.append(new_state)
        return neighbors
    
    def neighborhood3(state):
        """3-flip neighborhood"""
        neighbors = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    new_state = state.copy()
                    new_state[i] = not new_state[i]
                    new_state[j] = not new_state[j]
                    new_state[k] = not new_state[k]
                    neighbors.append(new_state)
        return neighbors
    
    neighborhoods = [neighborhood1, neighborhood2, neighborhood3]
    
    for iteration in range(max_iter):
        steps += 1
        improved = False
        
        # Try each neighborhood in order
        for neighborhood in neighborhoods:
            neighbors = neighborhood(current)
            total_nodes += len(neighbors)
            
            best_neighbor_val = current_val
            best_neighbor = None
            
            for neighbor in neighbors:
                neighbor_val = heuristic(clauses, neighbor)
                if neighbor_val < best_neighbor_val:
                    best_neighbor_val = neighbor_val
                    best_neighbor = neighbor
            
            # Move to better neighbor if found
            if best_neighbor is not None:
                current, current_val = best_neighbor, best_neighbor_val
                improved = True
                break  # Restart with first neighborhood
        
        if not improved:
            break  # No improvement in any neighborhood
            
        if current_val == 0:
            success = 1
            break
    
    penetrance = total_nodes / steps if steps > 0 else 0
    elapsed = time.time() - start
    return success, steps, total_nodes, penetrance, elapsed

def run_multiple_trials(n_vars, n_clauses, num_trials=10):
    """Run multiple trials to get reliable statistics"""
    all_results = []
    
    for trial in range(num_trials):
        clauses = generate_3sat(n_clauses, n_vars)
        trial_results = []
        
        for h_name, h_func in [("H1", h1_sat), ("H2", h2_unsat)]:
            for algo_name, algo_func in [
                ("Hill-Climbing", hill_climbing),
                ("Beam Search (w=3)", lambda c, n, h: beam_search(c, n, 3, h)),
                ("Beam Search (w=4)", lambda c, n, h: beam_search(c, n, 4, h)),
                ("VND", vnd)
            ]:
                success, steps, nodes, penetrance, elapsed = algo_func(clauses, n_vars, h_func)
                trial_results.append((algo_name, h_name, success, steps, nodes, elapsed, penetrance))
        
        all_results.append(trial_results)
    
    return all_results

def calculate_averages(all_results):
    """Calculate average performance across trials"""
    algo_stats = {}
    
    for trial_results in all_results:
        for algo_name, h_name, success, steps, nodes, elapsed, penetrance in trial_results:
            key = (algo_name, h_name)
            if key not in algo_stats:
                algo_stats[key] = {
                    'success_sum': 0, 'steps_sum': 0, 'nodes_sum': 0,
                    'time_sum': 0, 'penetrance_sum': 0, 'count': 0
                }
            
            stats = algo_stats[key]
            stats['success_sum'] += success
            stats['steps_sum'] += steps
            stats['nodes_sum'] += nodes
            stats['time_sum'] += elapsed
            stats['penetrance_sum'] += penetrance
            stats['count'] += 1
    
    # Calculate averages
    averages = {}
    for key, stats in algo_stats.items():
        count = stats['count']
        averages[key] = {
            'success_rate': stats['success_sum'] / count,
            'avg_steps': stats['steps_sum'] / count,
            'avg_nodes': stats['nodes_sum'] / count,
            'avg_time': stats['time_sum'] / count,
            'avg_penetrance': stats['penetrance_sum'] / count
        }
    
    return averages

if __name__ == "__main__":
    n_vars, n_clauses = 20, 90
    num_trials = 10
    
    print("3-SAT LOCAL SEARCH ALGORITHM COMPARISON")
    print(f"Configuration: n={n_vars}, m={n_clauses}, trials={num_trials}")
    print("=" * 75)
    print(f"{'Algorithm':<18} {'Heuristic':<6} {'Success':<7} {'Steps':<8} {'Nodes':<10} {'Time(s)':<8} {'Penetrance':<10}")
    print("-" * 75)
    
    # multiple trials 
    all_results = run_multiple_trials(n_vars, n_clauses, num_trials)
    averages = calculate_averages(all_results)
    
    # results
    for (algo_name, h_name), stats in averages.items():
        success_rate = stats['success_rate']
        avg_steps = stats['avg_steps']
        avg_nodes = stats['avg_nodes']
        avg_time = stats['avg_time']
        avg_penetrance = stats['avg_penetrance']
        
        print(f"{algo_name:<18} {h_name:<6} {success_rate:<7.2f} {avg_steps:<8.1f} "
              f"{avg_nodes:<10.0f} {avg_time:<8.4f} {avg_penetrance:<10.1f}")
    
    print("=" * 75)
    print("Note: Results based on average of multiple trials")
    print("Algorithms show natural performance variations")