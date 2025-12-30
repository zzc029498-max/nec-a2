import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict
import time
from copy import deepcopy

class Graph:
    """Represents a graph for the coloring problem"""
    def __init__(self, vertices: int):
        self.V = vertices
        self.edges = []
        self.adj_matrix = np.zeros((vertices, vertices), dtype=int)
    
    def add_edge(self, u: int, v: int):
        """Add an edge between vertices u and v"""
        self.edges.append((u, v))
        self.adj_matrix[u][v] = 1
        self.adj_matrix[v][u] = 1
    
    def get_neighbors(self, vertex: int) -> List[int]:
        """Get all neighbors of a vertex"""
        return [i for i in range(self.V) if self.adj_matrix[vertex][i] == 1]
    
    @staticmethod
    def load_from_file(filename: str):
        """Load graph from DIMACS format file"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        vertices = 0
        edges_list = []
        
        for line in lines:
            if line.startswith('p'):
                parts = line.split()
                vertices = int(parts[2])
            elif line.startswith('e'):
                parts = line.split()
                u, v = int(parts[1]) - 1, int(parts[2]) - 1  # Convert to 0-indexed
                edges_list.append((u, v))
        
        graph = Graph(vertices)
        for u, v in edges_list:
            graph.add_edge(u, v)
        
        return graph

class GeneticAlgorithm:
    """Genetic Algorithm for Graph Coloring Problem"""
    
    def __init__(self, graph: Graph, pop_size: int = 100, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1, max_colors: int = None):
        self.graph = graph
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = int(pop_size * elitism_rate)
        self.max_colors = max_colors if max_colors else graph.V
        
        self.population = []
        self.fitness_history = []
        self.best_solution = None
        self.best_fitness = float('-inf')
    
    def create_chromosome(self) -> List[int]:
        """Create a random chromosome (solution)"""
        # Start with fewer colors for better initial solutions
        max_initial_colors = min(self.max_colors, self.graph.V // 2 + 1)
        return [random.randint(0, max_initial_colors - 1) for _ in range(self.graph.V)]
    
    def initialize_population(self):
        """Create initial population"""
        self.population = [self.create_chromosome() for _ in range(self.pop_size)]
    
    def count_conflicts(self, chromosome: List[int]) -> int:
        """Count the number of edge conflicts (adjacent vertices with same color)"""
        conflicts = 0
        for u, v in self.graph.edges:
            if chromosome[u] == chromosome[v]:
                conflicts += 1
        return conflicts
    
    def count_colors(self, chromosome: List[int]) -> int:
        """Count the number of unique colors used"""
        return len(set(chromosome))
    
    def fitness(self, chromosome: List[int]) -> float:
        """
        Calculate fitness of a chromosome
        Higher fitness is better
        Prioritize: 1) No conflicts, 2) Fewer colors
        """
        conflicts = self.count_conflicts(chromosome)
        colors = self.count_colors(chromosome)
        
        # Heavily penalize conflicts
        if conflicts > 0:
            return -1000 * conflicts - colors
        else:
            # Valid solution: minimize colors
            return 1000 - colors
    
    def is_valid_solution(self, chromosome: List[int]) -> bool:
        """Check if chromosome is a valid solution (no conflicts)"""
        return self.count_conflicts(chromosome) == 0
    
    # SELECTION METHODS
    
    def selection_tournament(self, tournament_size: int = 3) -> List[int]:
        """Tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: self.fitness(x))
    
    def selection_roulette(self) -> List[int]:
        """Roulette wheel selection"""
        fitnesses = [self.fitness(ind) for ind in self.population]
        
        # Shift fitnesses to be positive
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1 for f in fitnesses]
        
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choice(self.population)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitnesses):
            current += fitness
            if current > pick:
                return self.population[i]
        
        return self.population[-1]
    
    def selection_rank(self) -> List[int]:
        """Rank-based selection"""
        sorted_pop = sorted(self.population, key=lambda x: self.fitness(x))
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)
        
        pick = random.uniform(0, total_rank)
        current = 0
        for i, rank in enumerate(ranks):
            current += rank
            if current > pick:
                return sorted_pop[i]
        
        return sorted_pop[-1]
    
    # CROSSOVER METHODS
    
    def crossover_single_point(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def crossover_two_point(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Two-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        
        points = sorted(random.sample(range(1, len(parent1)), 2))
        child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
        child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
        return child1, child2
    
    def crossover_uniform(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Uniform crossover"""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        
        child1, child2 = [], []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2
    
    # MUTATION METHODS
    
    def mutation_random(self, chromosome: List[int]) -> List[int]:
        """Random color change mutation"""
        mutated = chromosome[:]
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = random.randint(0, self.max_colors - 1)
        return mutated
    
    def mutation_swap(self, chromosome: List[int]) -> List[int]:
        """Swap colors mutation"""
        mutated = chromosome[:]
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def mutation_greedy_repair(self, chromosome: List[int]) -> List[int]:
        """Greedy mutation: fix conflicts by changing to a valid color"""
        mutated = chromosome[:]
        
        if random.random() < self.mutation_rate:
            # Find conflicting vertices
            conflicting = []
            for u, v in self.graph.edges:
                if mutated[u] == mutated[v]:
                    conflicting.extend([u, v])
            
            if conflicting:
                # Pick a random conflicting vertex
                vertex = random.choice(conflicting)
                
                # Find a color that doesn't conflict with neighbors
                neighbor_colors = {mutated[n] for n in self.graph.get_neighbors(vertex)}
                available_colors = [c for c in range(self.max_colors) if c not in neighbor_colors]
                
                if available_colors:
                    mutated[vertex] = random.choice(available_colors)
                else:
                    mutated[vertex] = random.randint(0, self.max_colors - 1)
        
        return mutated
    
    def evolve(self, generations: int = 1000, selection_method: str = 'tournament',
               crossover_method: str = 'single_point', mutation_method: str = 'random',
               verbose: bool = True, stagnation_limit: int = 100) -> Dict:
        """
        Run the genetic algorithm
        
        selection_method: 'tournament', 'roulette', 'rank'
        crossover_method: 'single_point', 'two_point', 'uniform'
        mutation_method: 'random', 'swap', 'greedy_repair'
        """
        
        start_time = time.time()
        self.initialize_population()
        self.fitness_history = []
        
        # Select methods
        selection_methods = {
            'tournament': self.selection_tournament,
            'roulette': self.selection_roulette,
            'rank': self.selection_rank
        }
        
        crossover_methods = {
            'single_point': self.crossover_single_point,
            'two_point': self.crossover_two_point,
            'uniform': self.crossover_uniform
        }
        
        mutation_methods = {
            'random': self.mutation_random,
            'swap': self.mutation_swap,
            'greedy_repair': self.mutation_greedy_repair
        }
        
        select = selection_methods[selection_method]
        crossover = crossover_methods[crossover_method]
        mutate = mutation_methods[mutation_method]
        
        stagnation_counter = 0
        prev_best_fitness = float('-inf')
        
        for gen in range(generations):
            # Evaluate fitness
            fitnesses = [(ind, self.fitness(ind)) for ind in self.population]
            fitnesses.sort(key=lambda x: x[1], reverse=True)
            
            # Track best solution
            best_ind, best_fit = fitnesses[0]
            self.fitness_history.append(best_fit)
            
            if best_fit > self.best_fitness:
                self.best_fitness = best_fit
                self.best_solution = best_ind[:]
            
            # Check for stagnation
            if best_fit == prev_best_fitness:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best_fitness = best_fit
            
            if verbose and gen % 100 == 0:
                colors = self.count_colors(best_ind)
                conflicts = self.count_conflicts(best_ind)
                print(f"Gen {gen}: Best Fitness = {best_fit:.2f}, Colors = {colors}, Conflicts = {conflicts}")
            
            # Early stopping
            if self.is_valid_solution(best_ind) and stagnation_counter > stagnation_limit:
                if verbose:
                    print(f"Converged at generation {gen}")
                break
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            new_population.extend([ind for ind, _ in fitnesses[:self.elitism_count]])
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1 = select()
                parent2 = select()
                
                child1, child2 = crossover(parent1, parent2)
                
                child1 = mutate(child1)
                child2 = mutate(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.pop_size]
        
        elapsed_time = time.time() - start_time
        
        # Final results
        final_colors = self.count_colors(self.best_solution)
        final_conflicts = self.count_conflicts(self.best_solution)
        
        results = {
            'best_solution': self.best_solution,
            'colors_used': final_colors,
            'conflicts': final_conflicts,
            'is_valid': final_conflicts == 0,
            'generations': len(self.fitness_history),
            'time': elapsed_time,
            'fitness_history': self.fitness_history
        }
        
        if verbose:
            print("\n" + "="*50)
            print("FINAL RESULTS")
            print("="*50)
            print(f"Colors used: {final_colors}")
            print(f"Conflicts: {final_conflicts}")
            print(f"Valid solution: {final_conflicts == 0}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print("="*50)
        
        return results

def plot_fitness_evolution(fitness_history: List[float], title: str = "Fitness Evolution"):
    """Plot the evolution of fitness over generations"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fitness_evolution.png', dpi=300)
    plt.show()

def create_sample_graph(size: str = 'small') -> Graph:
    """Create sample graphs for testing"""
    if size == 'small':
        # Small graph with 6 vertices
        graph = Graph(6)
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
        for u, v in edges:
            graph.add_edge(u, v)
    elif size == 'medium':
        # Medium graph with random edges
        n = 50
        graph = Graph(n)
        for _ in range(n * 3):  # Add 3*n random edges
            u, v = random.sample(range(n), 2)
            if graph.adj_matrix[u][v] == 0:
                graph.add_edge(u, v)
    else:  # large
        n = 100
        graph = Graph(n)
        for _ in range(n * 4):
            u, v = random.sample(range(n), 2)
            if graph.adj_matrix[u][v] == 0:
                graph.add_edge(u, v)
    
    return graph

# EXAMPLE USAGE
if __name__ == "__main__":
    print("Graph Coloring Problem - Genetic Algorithm")
    print("="*50)
    
    # Create a sample graph
    # graph = create_sample_graph('small')
    graph = Graph.load_from_file("/Users/jiangyutang/Desktop/a2/queen7_7.col") 
    print(f"Graph: {graph.V} vertices, {len(graph.edges)} edges")
    
    # Test different parameter combinations
    configs = [
        {'selection': 'tournament', 'crossover': 'single_point', 'mutation': 'random'},
        {'selection': 'tournament', 'crossover': 'two_point', 'mutation': 'greedy_repair'},
        {'selection': 'roulette', 'crossover': 'uniform', 'mutation': 'random'},
        {'selection': 'rank', 'crossover': 'single_point', 'mutation': 'greedy_repair'},
        {'selection': 'tournament', 'crossover': 'uniform', 'mutation': 'swap'},
        {'selection': 'roulette', 'crossover': 'two_point', 'mutation': 'greedy_repair'},
    ]
    
    print("\nTesting different configurations...")
    print("-"*50)
    
    results_list = []
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config}")
        ga = GeneticAlgorithm(graph, pop_size=100, mutation_rate=0.1, 
                             crossover_rate=0.8, elitism_rate=0.1)
        
        results = ga.evolve(
            generations=500,
            selection_method=config['selection'],
            crossover_method=config['crossover'],
            mutation_method=config['mutation'],
            verbose=False
        )
        
        results_list.append((config, results))
        print(f"  → Colors: {results['colors_used']}, Valid: {results['is_valid']}, Time: {results['time']:.2f}s")
    
    # Find and display best result
    best_config, best_results = min(results_list, 
                                    key=lambda x: (x[1]['conflicts'], x[1]['colors_used']))
    
    print("\n" + "="*50)
    print("BEST CONFIGURATION")
    print("="*50)
    print(f"Config: {best_config}")
    print(f"Colors: {best_results['colors_used']}")
    print(f"Conflicts: {best_results['conflicts']}")
    print(f"Valid: {best_results['is_valid']}")
    print("="*50)
    
    # Plot fitness evolution of best result
    plot_fitness_evolution(best_results['fitness_history'], 
                          title=f"Best Configuration - {best_results['colors_used']} colors")