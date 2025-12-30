import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict
import time
from copy import deepcopy
import math

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
                u, v = int(parts[1]) - 1, int(parts[2]) - 1
                edges_list.append((u, v))
        
        graph = Graph(vertices)
        for u, v in edges_list:
            graph.add_edge(u, v)
        
        return graph

class BaseOptimizer:
    """Base class for optimization algorithms"""
    
    def __init__(self, graph: Graph, max_colors: int = None):
        self.graph = graph
        self.max_colors = max_colors if max_colors else graph.V
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def count_conflicts(self, solution: List[int]) -> int:
        """Count the number of edge conflicts"""
        conflicts = 0
        for u, v in self.graph.edges:
            if solution[u] == solution[v]:
                conflicts += 1
        return conflicts
    
    def count_colors(self, solution: List[int]) -> int:
        """Count the number of unique colors used"""
        return len(set(solution))
    
    def fitness(self, solution: List[int]) -> float:
        """Calculate fitness of a solution"""
        conflicts = self.count_conflicts(solution)
        colors = self.count_colors(solution)
        
        if conflicts > 0:
            return -1000 * conflicts - colors
        else:
            return 1000 - colors
    
    def is_valid_solution(self, solution: List[int]) -> bool:
        """Check if solution is valid (no conflicts)"""
        return self.count_conflicts(solution) == 0
    
    def create_random_solution(self) -> List[int]:
        """Create a random solution"""
        max_initial_colors = min(self.max_colors, self.graph.V // 2 + 1)
        return [random.randint(0, max_initial_colors - 1) for _ in range(self.graph.V)]
    
    def create_greedy_solution(self) -> List[int]:
        """Create a greedy initial solution"""
        solution = [-1] * self.graph.V
        
        for vertex in range(self.graph.V):
            neighbor_colors = {solution[n] for n in self.graph.get_neighbors(vertex) if solution[n] != -1}
            
            for color in range(self.max_colors):
                if color not in neighbor_colors:
                    solution[vertex] = color
                    break
        
        return solution

class GeneticAlgorithm(BaseOptimizer):
    """Genetic Algorithm for Graph Coloring Problem"""
    
    def __init__(self, graph: Graph, pop_size: int = 100, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1, max_colors: int = None):
        super().__init__(graph, max_colors)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = int(pop_size * elitism_rate)
        self.population = []
    
    def initialize_population(self):
        """Create initial population"""
        self.population = [self.create_random_solution() for _ in range(self.pop_size)]
    
    def selection_tournament(self, tournament_size: int = 3) -> List[int]:
        """Tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: self.fitness(x))
    
    def selection_roulette(self) -> List[int]:
        """Roulette wheel selection"""
        fitnesses = [self.fitness(ind) for ind in self.population]
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
        """Greedy mutation: fix conflicts"""
        mutated = chromosome[:]
        
        if random.random() < self.mutation_rate:
            conflicting = []
            for u, v in self.graph.edges:
                if mutated[u] == mutated[v]:
                    conflicting.extend([u, v])
            
            if conflicting:
                vertex = random.choice(conflicting)
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
        """Run the genetic algorithm"""
        
        start_time = time.time()
        self.initialize_population()
        self.fitness_history = []
        
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
            fitnesses = [(ind, self.fitness(ind)) for ind in self.population]
            fitnesses.sort(key=lambda x: x[1], reverse=True)
            
            best_ind, best_fit = fitnesses[0]
            self.fitness_history.append(best_fit)
            
            if best_fit > self.best_fitness:
                self.best_fitness = best_fit
                self.best_solution = best_ind[:]
            
            if best_fit == prev_best_fitness:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best_fitness = best_fit
            
            if verbose and gen % 100 == 0:
                colors = self.count_colors(best_ind)
                conflicts = self.count_conflicts(best_ind)
                print(f"Gen {gen}: Fitness = {best_fit:.2f}, Colors = {colors}, Conflicts = {conflicts}")
            
            if self.is_valid_solution(best_ind) and stagnation_counter > stagnation_limit:
                if verbose:
                    print(f"Converged at generation {gen}")
                break
            
            new_population = []
            new_population.extend([ind for ind, _ in fitnesses[:self.elitism_count]])
            
            while len(new_population) < self.pop_size:
                parent1 = select()
                parent2 = select()
                
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1)
                child2 = mutate(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.pop_size]
        
        elapsed_time = time.time() - start_time
        
        return {
            'algorithm': 'Genetic Algorithm',
            'best_solution': self.best_solution,
            'colors_used': self.count_colors(self.best_solution),
            'conflicts': self.count_conflicts(self.best_solution),
            'is_valid': self.is_valid_solution(self.best_solution),
            'generations': len(self.fitness_history),
            'time': elapsed_time,
            'fitness_history': self.fitness_history
        }

class SimulatedAnnealing(BaseOptimizer):
    """Simulated Annealing for Graph Coloring Problem"""
    
    def __init__(self, graph: Graph, initial_temp: float = 100.0,
                 cooling_rate: float = 0.995, min_temp: float = 0.01,
                 max_colors: int = None):
        super().__init__(graph, max_colors)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
    
    def get_neighbor(self, solution: List[int]) -> List[int]:
        """Generate a neighboring solution by changing one vertex color"""
        neighbor = solution[:]
        vertex = random.randint(0, len(solution) - 1)
        
        # Try to pick a color that reduces conflicts
        if random.random() < 0.7:  # 70% greedy, 30% random
            neighbor_colors = {neighbor[n] for n in self.graph.get_neighbors(vertex)}
            available = [c for c in range(self.max_colors) if c not in neighbor_colors]
            if available:
                neighbor[vertex] = random.choice(available)
            else:
                neighbor[vertex] = random.randint(0, self.max_colors - 1)
        else:
            neighbor[vertex] = random.randint(0, self.max_colors - 1)
        
        return neighbor
    
    def optimize(self, max_iterations: int = 10000, verbose: bool = True) -> Dict:
        """Run simulated annealing"""
        
        start_time = time.time()
        
        # Initialize with greedy solution
        current_solution = self.create_greedy_solution()
        current_fitness = self.fitness(current_solution)
        
        self.best_solution = current_solution[:]
        self.best_fitness = current_fitness
        self.fitness_history = [current_fitness]
        
        temperature = self.initial_temp
        iterations = 0
        
        while temperature > self.min_temp and iterations < max_iterations:
            # Generate neighbor
            neighbor = self.get_neighbor(current_solution)
            neighbor_fitness = self.fitness(neighbor)
            
            # Calculate acceptance probability
            delta = neighbor_fitness - current_fitness
            
            if delta > 0:
                # Better solution, always accept
                current_solution = neighbor
                current_fitness = neighbor_fitness
            else:
                # Worse solution, accept with probability
                acceptance_prob = math.exp(delta / temperature)
                if random.random() < acceptance_prob:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
            
            # Update best solution
            if current_fitness > self.best_fitness:
                self.best_solution = current_solution[:]
                self.best_fitness = current_fitness
            
            self.fitness_history.append(self.best_fitness)
            
            # Cool down
            temperature *= self.cooling_rate
            iterations += 1
            
            if verbose and iterations % 1000 == 0:
                colors = self.count_colors(self.best_solution)
                conflicts = self.count_conflicts(self.best_solution)
                print(f"Iter {iterations}: Temp = {temperature:.4f}, Colors = {colors}, Conflicts = {conflicts}")
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nCompleted {iterations} iterations")
        
        return {
            'algorithm': 'Simulated Annealing',
            'best_solution': self.best_solution,
            'colors_used': self.count_colors(self.best_solution),
            'conflicts': self.count_conflicts(self.best_solution),
            'is_valid': self.is_valid_solution(self.best_solution),
            'iterations': iterations,
            'time': elapsed_time,
            'fitness_history': self.fitness_history
        }

class TabuSearch(BaseOptimizer):
    """Tabu Search for Graph Coloring Problem"""
    
    def __init__(self, graph: Graph, tabu_tenure: int = 10,
                 max_colors: int = None):
        super().__init__(graph, max_colors)
        self.tabu_tenure = tabu_tenure
        self.tabu_list = []
    
    def get_neighbors(self, solution: List[int], num_neighbors: int = 20) -> List[List[int]]:
        """Generate multiple neighboring solutions"""
        neighbors = []
        
        for _ in range(num_neighbors):
            neighbor = solution[:]
            vertex = random.randint(0, len(solution) - 1)
            
            # Prefer colors that reduce conflicts
            neighbor_colors = {neighbor[n] for n in self.graph.get_neighbors(vertex)}
            available = [c for c in range(self.max_colors) if c not in neighbor_colors and c != neighbor[vertex]]
            
            if available:
                neighbor[vertex] = random.choice(available)
            else:
                new_color = random.randint(0, self.max_colors - 1)
                if new_color != neighbor[vertex]:
                    neighbor[vertex] = new_color
                else:
                    continue
            
            neighbors.append(neighbor)
        
        return neighbors
    
    def is_tabu(self, solution: List[int]) -> bool:
        """Check if solution is in tabu list"""
        solution_tuple = tuple(solution)
        return solution_tuple in self.tabu_list
    
    def add_to_tabu(self, solution: List[int]):
        """Add solution to tabu list"""
        solution_tuple = tuple(solution)
        self.tabu_list.append(solution_tuple)
        
        # Maintain tabu list size
        if len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.pop(0)
    
    def optimize(self, max_iterations: int = 1000, verbose: bool = True) -> Dict:
        """Run tabu search"""
        
        start_time = time.time()
        
        # Initialize with greedy solution
        current_solution = self.create_greedy_solution()
        current_fitness = self.fitness(current_solution)
        
        self.best_solution = current_solution[:]
        self.best_fitness = current_fitness
        self.fitness_history = [current_fitness]
        
        stagnation_counter = 0
        stagnation_limit = 100
        
        for iteration in range(max_iterations):
            # Generate neighbors
            neighbors = self.get_neighbors(current_solution, num_neighbors=30)
            
            # Find best non-tabu neighbor (or best tabu if it's better than best_solution - aspiration criteria)
            best_neighbor = None
            best_neighbor_fitness = float('-inf')
            
            for neighbor in neighbors:
                neighbor_fitness = self.fitness(neighbor)
                
                # Aspiration criteria: accept tabu if better than best known
                if neighbor_fitness > self.best_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    break
                
                # Accept if not tabu and better than current best neighbor
                if not self.is_tabu(neighbor) and neighbor_fitness > best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
            
            # If no valid neighbor found, restart with random solution
            if best_neighbor is None:
                current_solution = self.create_random_solution()
                current_fitness = self.fitness(current_solution)
                self.tabu_list.clear()
                continue
            
            # Move to best neighbor
            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness
            self.add_to_tabu(current_solution)
            
            # Update best solution
            if current_fitness > self.best_fitness:
                self.best_solution = current_solution[:]
                self.best_fitness = current_fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            self.fitness_history.append(self.best_fitness)
            
            if verbose and iteration % 100 == 0:
                colors = self.count_colors(self.best_solution)
                conflicts = self.count_conflicts(self.best_solution)
                print(f"Iter {iteration}: Colors = {colors}, Conflicts = {conflicts}, Tabu size = {len(self.tabu_list)}")
            
            # Early stopping
            if self.is_valid_solution(self.best_solution) and stagnation_counter > stagnation_limit:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        elapsed_time = time.time() - start_time
        
        return {
            'algorithm': 'Tabu Search',
            'best_solution': self.best_solution,
            'colors_used': self.count_colors(self.best_solution),
            'conflicts': self.count_conflicts(self.best_solution),
            'is_valid': self.is_valid_solution(self.best_solution),
            'iterations': len(self.fitness_history),
            'time': elapsed_time,
            'fitness_history': self.fitness_history
        }

def plot_fitness_evolution(fitness_history: List[float], title: str = "Fitness Evolution"):
    """Plot the evolution of fitness"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'fitness_{title.replace(" ", "_")}.png', dpi=300)
    plt.show()

def compare_algorithms(graph: Graph, algorithms: List[Tuple], verbose: bool = True):
    """Compare multiple optimization algorithms"""
    
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON")
    print("="*70)
    
    results = []
    
    for name, optimizer, params in algorithms:
        print(f"\nRunning {name}...")
        print("-"*70)
        
        result = optimizer.optimize(**params) if hasattr(optimizer, 'optimize') else optimizer.evolve(**params)
        results.append((name, result))
        
        print(f"\n{name} Results:")
        print(f"  Colors: {result['colors_used']}")
        print(f"  Conflicts: {result['conflicts']}")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Time: {result['time']:.2f}s")
        print(f"  Iterations/Generations: {result.get('iterations', result.get('generations', 0))}")
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Algorithm':<25} {'Colors':<10} {'Valid':<10} {'Time (s)':<10}")
    print("-"*70)
    
    for name, result in results:
        print(f"{name:<25} {result['colors_used']:<10} {str(result['is_valid']):<10} {result['time']:<10.2f}")
    
    print("="*70)
    
    # Plot all fitness evolutions
    plt.figure(figsize=(12, 6))
    for name, result in results:
        plt.plot(result['fitness_history'], label=name, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Algorithm Comparison - Fitness Evolution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300)
    plt.show()
    
    return results

def create_sample_graph(size: str = 'small') -> Graph:
    """Create sample graphs for testing"""
    if size == 'small':
        graph = Graph(6)
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
        for u, v in edges:
            graph.add_edge(u, v)
    elif size == 'medium':
        n = 50
        graph = Graph(n)
        for _ in range(n * 3):
            u, v = random.sample(range(n), 2)
            if graph.adj_matrix[u][v] == 0:
                graph.add_edge(u, v)
    else:
        n = 100
        graph = Graph(n)
        for _ in range(n * 4):
            u, v = random.sample(range(n), 2)
            if graph.adj_matrix[u][v] == 0:
                graph.add_edge(u, v)
    
    return graph

# EXAMPLE USAGE
if __name__ == "__main__":
    print("Graph Coloring Problem - Multi-Algorithm Comparison")
    print("="*70)
    
    # Create a sample graph
    graph = Graph.load_from_file("/Users/jiangyutang/Desktop/a2/queen7_7.col") 

    print(f"Graph: {graph.V} vertices, {len(graph.edges)} edges\n")
    
    # Setup algorithms
    ga = GeneticAlgorithm(graph, pop_size=100, mutation_rate=0.1, crossover_rate=0.8)
    sa = SimulatedAnnealing(graph, initial_temp=100.0, cooling_rate=0.995)
    ts = TabuSearch(graph, tabu_tenure=10)
    
    algorithms = [
        ("Genetic Algorithm", ga, {
            'generations': 500,
            'selection_method': 'tournament',
            'crossover_method': 'two_point',
            'mutation_method': 'greedy_repair',
            'verbose': False
        }),
        ("Simulated Annealing", sa, {
            'max_iterations': 5000,
            'verbose': False
        }),
        ("Tabu Search", ts, {
            'max_iterations': 500,
            'verbose': False
        })
    ]
    
    # Run comparison
    results = compare_algorithms(graph, algorithms, verbose=True)
    
    print("\n✓ All algorithms completed!")
    print("✓ Comparison plot saved as 'algorithm_comparison.png'")