import math
import numpy as np
import tsplib95

# Class representing the environment of the ant colony
"""
    rho: pheromone evaporation rate
"""


class Environment:
    def __init__(self, rho: float):
        if not 0 <= rho <= 1:
            raise ValueError("Evaporation rate (rho) must be between 0 and 1")

        self.rho = rho

        # Initialize the environment topology
        self.graph = tsplib95.load_problem("att48-specs/att48.tsp").get_graph()

        # Initialize the pheromone map in the environment
        self.num_vertices = self.graph.number_of_nodes()
        self.initialize_pheromone_map()

    # Initialize the pheromone trails in the environment
    def initialize_pheromone_map(self):
        # Select random start vertex
        start_vertex = np.random.randint(1, self.num_vertices + 1)
        current_vertex = start_vertex
        visited = [current_vertex]
        distance = 0

        # Build tour by selecting nearest unvisited vertex
        while len(visited) < self.num_vertices:
            # Find nearest unvisited neighbor
            nearest = min(
                (v for v in range(1, self.num_vertices + 1) if v not in visited),
                key=lambda x: self.graph[current_vertex][x]["weight"]
            )
            distance += self.graph[current_vertex][nearest]["weight"]
            current_vertex = nearest
            visited.append(current_vertex)

        # Complete the tour
        distance += self.graph[current_vertex][start_vertex]["weight"]

        # Initialize pheromone levels
        initial_pheromone = 48 / distance
        for i, j in self.graph.edges():
            self.graph[i][j]["pheromone_level"] = initial_pheromone

    # Update the pheromone trails in the environment
    def update_pheromone_map(self, ants: list):
        # Evaporate pheromones
        for i, j in self.graph.edges():
            self.graph[i][j]["pheromone_level"] *= (1 - self.rho)

        # Add new pheromones
        for ant in ants:
            deposit = 1 / ant.travelled_distance
            for i in range(len(ant.tour) - 1):
                self.graph[ant.tour[i]][ant.tour[i + 1]]["pheromone_level"] += deposit

    # Get the pheromone trails in the environment
    def get_pheromone_map(self):
        pheromone_map = {}
        for i, j in self.graph.edges():
            pheromone_map[(i,j)] = self.graph[i][j]["pheromone_level"]
        return pheromone_map

    # Get the environment topology
    def get_possible_locations(self):
        return list(self.graph.nodes())
