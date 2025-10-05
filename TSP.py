import numpy as np
import random
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

def haversine(coord1, coord2):
    R = 6371.0  # Earth radius in km
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

locations = {
    "Jaipur": (26.9124, 75.7873),
    "Udaipur": (24.5854, 73.7125),
    "Jodhpur": (26.2389, 73.0243),
    "Ajmer": (26.4499, 74.6399),
    "Jaisalmer": (26.9157, 70.9083),
    "Bikaner": (28.0229, 73.3119),
    "Mount Abu": (24.5926, 72.7156),
    "Pushkar": (26.4899, 74.5521),
    "Bharatpur": (27.2176, 77.4895),
    "Kota": (25.2138, 75.8648),
    "Chittorgarh": (24.8887, 74.6269),
    "Alwar": (27.5665, 76.6250),
    "Ranthambore": (26.0173, 76.5026),
    "Sariska": (27.3309, 76.4154),
    "Mandawa": (28.0524, 75.1416),
    "Dungarpur": (23.8430, 73.7142),
    "Bundi": (25.4305, 75.6499),
    "Sikar": (27.6094, 75.1399),
    "Nagaur": (27.2020, 73.7336),
    "Shekhawati": (27.6485, 75.5455),
}

N = len(locations)
cities = list(locations.keys())
D = np.zeros((N, N))

for i in range(N):
    for j in range(i + 1, N):
        dist = haversine(locations[cities[i]], locations[cities[j]])
        D[i, j] = dist
        D[j, i] = dist

def path_cost_tour(tour, distance_matrix):
    cost = 0
    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i], tour[i + 1]]
    cost += distance_matrix[tour[-1], tour[0]]
    return cost

def generate_neighbor(tour):
    i, j = sorted(random.sample(range(N), 2))
    new_tour = tour.copy()
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]  # Swap
    return new_tour

def simulated_annealing(distance_matrix, initial_temp=1000, alpha=0.995, max_outer=1000):
    N = len(distance_matrix)
    current_tour = list(range(N))  # Sequential initial for reproducibility, or random.sample(range(N), N)
    random.shuffle(current_tour)
    current_cost = path_cost_tour(current_tour, distance_matrix)
    best_tour = current_tour[:]
    best_cost = current_cost
    print("Initial Cost:", current_cost)  # For sync with LaTeX example

    cost_history = [current_cost]
    temperature = initial_temp

    for outer in range(max_outer):
        for inner in range(N):  # Attempts per temperature level
            new_tour = generate_neighbor(current_tour)
            new_cost = path_cost_tour(new_tour, distance_matrix)
            delta_cost = new_cost - current_cost

            if delta_cost < 0 or random.random() < np.exp(-delta_cost / temperature):
                current_tour = new_tour
                current_cost = new_cost

            if current_cost < best_cost:
                best_tour = current_tour[:]
                best_cost = current_cost

            cost_history.append(best_cost)

        temperature *= alpha
        if temperature < 1e-3:
            break

    return best_tour, best_cost, cost_history

best_tour, best_cost, cost_history = simulated_annealing(D)

print("Best Tour:", [cities[i] for i in best_tour])
print("Best Cost:", best_cost)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
tour_coords = np.array(
    [locations[cities[i]] for i in best_tour] + [locations[cities[best_tour[0]]]]
)
plt.plot(tour_coords[:, 1], tour_coords[:, 0], "o-", label="Optimized Tour")
plt.title("Optimized Tour")
for i, city in enumerate(best_tour):
    plt.text(tour_coords[i, 1] + 0.05, tour_coords[i, 0] + 0.05, cities[city], fontsize=8)

plt.subplot(1, 2, 2)
plt.plot(cost_history)
plt.title("Tour Cost Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Tour Cost")
plt.show()

'''
OUTPUT:
Initial Cost: 5763.336447723138
Best Tour: ['Nagaur', 'Mandawa', 'Shekhawati', 'Sikar', 'Pushkar', 'Ajmer', 'Jaipur', 'Sariska', 'Alwar', 'Bharatpur', 'Ranthambore', 'Kota', 'Bundi', 'Chittorgarh', 'Udaipur', 'Dungarpur', 'Mount Abu', 'Jodhpur', 'Jaisalmer', 'Bikaner']
Best Cost: 2260.8877563694186
'''