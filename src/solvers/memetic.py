import time
import numpy as np

from datamodels.problem import Problem
from optimization import Optimization
from utils.log import log

POSITIVE_INFINITY = float("inf")

class MemeticAlgorithm:
    def __init__(
        self,
        file_name: str,
        problem: Problem,
        pop_size: int,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        time_limit: int = 60 * 5,  # seconds
        tol: int = 1e-6,
    ):        
        self.file_name = file_name
        self.problem = problem
        self.pop_size = pop_size
        self.fitness = POSITIVE_INFINITY * np.ones(pop_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.time_limit = time_limit
        self.tol = tol
        self.optimization = Optimization(problem)
        self.pop = None

    # ==========================
    # Memetic Algorithm Functions
    # ==========================

    def _init_populaton(self, pop_size) -> np.ndarray:
        pop = [
                [
                    np.random.randint(1, self.problem.time_horizon.time_steps) 
                    for _ in range(len(self.problem.interventions))
                ] 
               for _ in range(pop_size)
        ]
        return np.array(pop)
    
    def _repair_individual(self, individual) -> np.ndarray:
        return individual

    def _select_parents(self, strategy="tournament"):
        """
        Seleciona dois pais. Estratégias possíveis:
        - random: seleciona dois pais aleatoriamente
        - tournament: seleciona dois pais através de torneio    
        - roulette: seleciona dois pais através de roleta
        """
        if strategy == "roulette":            
            total = self.fitness.sum()
            if total == 0:
                idx = np.random.sample(self.pop, 2)
            else:
                probs = self.fitness / total
                idx = np.random.choice(self.pop, size=2, replace=False, p=probs)

        elif strategy == "tournament":
            def tournament_selection(k=3):
                contenders = np.random.sample(self.pop, k)
                return max(contenders, key=lambda ind: self.fitness[ind])
            
            p1 = tournament_selection(self.pop)
            p2 = tournament_selection(self.pop)
            idx = [p1, p2]
        else:
            idx = np.random.sample(self.pop, 2)

        return self.pop[idx[0]], self.pop[idx[1]]
    
    def _crossover(self, parent_a, parent_b, strategy="one_point"):
        """
        Realiza o crossover entre dois pais para gerar um filho. Estratégias possíveis:
        - one_point: crossover de um ponto
        - two_point: crossover de dois pontos
        - uniform: crossover uniforme
        """
        n = len(parent_a)
        if strategy == "two_point":
            c1, c2 = sorted(np.random.choice(range(1, n), 2, replace=False))
            child = np.concatenate((parent_a[:c1], parent_b[c1:c2], parent_a[c2:]))
        elif strategy == "uniform":
            mask = np.random.rand(n) < 0.5
            child = np.where(mask, parent_a, parent_b)
        else: 
            c = np.random.randint(1, n)
            child = np.concatenate((parent_a[:c], parent_b[c:]), axis=0)
        return child

    def _mutate(self, individual):
        mutation_idx = np.random.randint(0, len(individual))
        individual[mutation_idx] = np.random.randint(
            1, self.problem.time_horizon.time_steps
        )
        return individual
    
    def _evaluate_individual(self, individual) -> float:
        _, penalty = self.optimization.constraints_satisfied(individual.tolist())
        fitness, _, _ = self.optimization.objective_function(individual, penalty)[0]
        return fitness
    

    def _local_search(self, individual) -> np.ndarray:
        return individual

    def _generate_new_population(self, pop):
        new_pop = []
        n_children = int(self.pop_size * self.crossover_rate)

        for _ in range(n_children):
            parent_a, parent_b = self._select_parents(pop)
            child = self._crossover(parent_a, parent_b)            
            child = self._local_search(child)

            if np.random.rand() < self.mutation_rate:
                child = self._mutate(child)

            child = self._local_search(child)
            new_pop.append(child)

        return np.array(new_pop)
    
    def update_populaton(self, pop, new_pop, strategy="generational"):
        """
        Estratégias possveis de atualização da população:
        - generational: substitui toda a população pela nova população gerada
        - steady_state: substitui apenas os piores indivíduos da população atual pelos melhores da nova população
        - elitism: mantém o melhor indivíduo da população atual e substitui o restante pela nova população gerada
        """
        if strategy == "generational":
            return new_pop

        elif strategy == "elitism":
            best_idx = np.argmin(self.fitness)
            best_individual = pop[best_idx]
            new_pop[0] = best_individual
            return new_pop

        elif strategy == "steady_state":
            combined = np.vstack((pop, new_pop))
            fitness_combined = np.array(
                [self._evaluate_individual(ind) for ind in combined]
            )
            best_indices = np.argsort(fitness_combined)[: self.pop_size]
            return combined[best_indices]

    def restart_population(self, pop, strategy="random"):
        """
        Reinicia a população para evitar convergência prematura. Estratégas possíveis:
        - random: gera uma nova população aleatória
        - elitista: mantém os melhores indivíduos e reinicia o restante
        """
        if strategy == "random":
            return self._init_populaton(self.pop_size)

        elif strategy == "elitist":
            percentage_to_keep = 0.2
            n_to_keep = int(self.pop_size * percentage_to_keep)

            best_idxs = np.argsort(self.fitness)[: n_to_keep]
            best_individuals = pop[best_idxs]
            new_pop = self._init_populaton(self.pop_size - n_to_keep)
            return new_pop + best_individuals
        

    def _has_converged(self) -> bool:
        return np.all(np.abs(self.fitness - self.fitness.mean()) < self.tol)

    # ==========================
    # Loop principal de otimização
    # ==========================

    def optimize(self) -> float:
        start_time = time.time()

        self.pop = self._init_populaton(self.pop_size)
        self.fitness = np.array(
            [self._evaluate_individual(ind) for ind in self.pop]
        )

        elapsed_time = time.time() - start_time
        while elapsed_time < self.time_limit:
            log(f"{self.file_name}", f"GA - Elapsed time: {elapsed_time:.2f} seconds.")                
            
            new_pop = self._generate_new_population(self.pop)
            self.pop = self.update_populaton(self.pop, new_pop, strategy="generational")
            self.fitness = np.array(
                [self._evaluate_individual(ind) for ind in self.pop]
            )

            if self._has_converged():
                log(self.file_name, "GA - Convergência atingida. Reiniciando população.")
                self.pop = self.restart_population(self.pop, strategy="random")
                self.fitness = np.array(
                    [self._evaluate_individual(ind) for ind in self.pop]
                )
            
            elapsed_time = time.time() - start_time

        
        best_idx = np.argmin(self.fitness)
        best_individual = self.pop[best_idx]
        best_fitness = self.fitness[best_idx]

        log(self.file_name, f"GA - Melhor indivíduo: {best_individual}")
        log(self.file_name, f"GA - Melhor fitness: {best_fitness:.6f}")

        return best_individual, best_fitness
