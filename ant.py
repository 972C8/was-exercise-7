
import numpy as np
import random

# Class representing an artificial ant of the ant colony
"""
    alpha: a parameter controlling the influence of the amount of pheromone during ants' path selection process
    beta: a parameter controlling the influence of the distance to the next node during ants' path selection process
"""


class Ant():
    def __init__(self, alpha: float, beta: float, initial_location):
        self.alpha = alpha
        self.beta = beta
        self.current_location = initial_location
        self.traveled_distance = 0
        self.reset()

    def reset(self):
        """Reset ant to initial state with random starting location."""
        self.current_location = np.int64(random.randint(1, 48))
        self.tour = [self.current_location]
        self.travelled_distance = 0

    # The ant runs to visit all the possible locations of the environment
    def run(self):
        """Execute ant's tour of all cities."""
        num_cities = self.environment.graph.number_of_nodes()

        # Visit all cities
        while len(self.tour) < num_cities:
            next_city = self.select_path()
            self.travelled_distance += self.get_distance(
                self.current_location, next_city)
            self.tour.append(next_city)
            self.current_location = next_city

        # Return to start
        self.tour.append(self.tour[0])
        self.travelled_distance += self.get_distance(
            self.current_location, self.tour[0])
        self.current_location = self.tour[0]

    def select_path(self):
        """Select next city using ACO probability rules."""
        # Get unvisited neighbors
        unvisited_neighbors = {
            city: self.environment.graph[self.current_location][city]
            for city in self.environment.graph[self.current_location]
            if city not in self.tour
        }

        if not unvisited_neighbors:
            return self.tour[0]  # Return to start if no unvisited neighbors

        # Calculate probabilities
        total = 0
        probabilities = []
        cities = list(unvisited_neighbors.keys())

        for city in cities:
            pheromone = unvisited_neighbors[city]["pheromone_level"]
            distance = unvisited_neighbors[city]["weight"]
            probability = (pheromone ** self.alpha) * \
                ((1 / distance) ** self.beta)
            probabilities.append(probability)
            total += probability

        # Normalize probabilities
        probabilities = [p/total for p in probabilities]

        # Select next city
        return np.random.choice(cities, p=probabilities)

    # Position an ant in an environment
    def join(self, environment):
        self.environment = environment

    def get_distance(self, from_vertex: np.int64, to_vertex: np.int64) -> float:
        """Calculate pseudo-Euclidean distance between two vertices.

        Args:
            from_vertex (int): Starting vertex ID
            to_vertex (int): Destination vertex ID

        Returns:
            float: Pseudo-Euclidean distance between the vertices
        """
        try:
            # Get distance directly from graph edge weight
            return self.environment.graph[from_vertex][to_vertex]['weight']
        except KeyError:
            # If edge doesn't exist, raise informative error
            raise ValueError(
                f"No path exists between vertices {from_vertex} and {to_vertex}")
