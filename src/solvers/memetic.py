import time
import numpy as np

from datamodels.problem import Problem
from solvers.optimization import Optimization
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
                    np.random.randint(1, self.problem.interventions[i].tmax + 1) # Garante que o tempo de início seja válido
                    for i in range(len(self.problem.interventions))
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
                idx = np.random.choice(range(0, self.pop_size), 2)
            else:
                probs = self.fitness / total
                idx = np.random.choice(range(0, self.pop_size), size=2, replace=False, p=probs)

        elif strategy == "tournament":
            def tournament_selection(k=3):
                contenders = np.random.choice(range(0, self.pop_size), k)
                return max(contenders, key=lambda ind: self.fitness[ind])
            
            p1 = tournament_selection()
            p2 = tournament_selection()
            idx = [p1, p2]
        else:
            idx = np.random.choice(range(0, self.pop_size), 2)

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
        fitness, _, _ = self.optimization.objective_function(individual, penalty)
        return fitness
    

    def _local_search(self, individual) -> np.ndarray:
        return individual

    def _generate_new_population(self):
        new_pop = []

        while len(new_pop) < self.pop_size:
            parent_a, parent_b = self._select_parents(strategy="roulette")
            if np.random.rand() > self.crossover_rate:
                child = self._crossover(parent_a, parent_b, strategy="uniform")            
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
        - elitism: mantém o melhor indivíduo da população atual e substitui o restante pela nova população gerada
        - steady_state: substitui apenas os piores indivíduos da população atual pelos melhores da nova população
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

    def restart_population(self, pop, strategy="elitist"):
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
            return np.vstack((new_pop, best_individuals))
        

    def _has_converged(self) -> bool:
        return np.all(np.abs(self.fitness - self.fitness.mean()) < self.tol)

    # ==========================
    # Loop principal de otimização
    # ==========================

    def optimize(self) -> tuple[np.ndarray, float]:
        start_time = time.time()
        # np.random.seed(start_time)

        self.pop = self._init_populaton(self.pop_size)
        self.fitness = np.array(
            [self._evaluate_individual(ind) for ind in self.pop]
        )

        elapsed_time = time.time() - start_time
        while elapsed_time < self.time_limit:
            log(f"{self.file_name}", f"Elapsed time: {elapsed_time:.2f} seconds.")                
            
            new_pop = self._generate_new_population()
            self.pop = self.update_populaton(self.pop, new_pop, strategy="steady_state")
            self.fitness = np.array(
                [self._evaluate_individual(ind) for ind in self.pop]
            )

            if self._has_converged():
                log(self.file_name, "Convergência atingida. Reiniciando população.")
                self.pop = self.restart_population(self.pop, strategy="elitist")
                self.fitness = np.array(
                    [self._evaluate_individual(ind) for ind in self.pop]
                )            
        
            best_idx = np.argmin(self.fitness)
            best_individual = self.pop[best_idx]
            best_fitness = self.fitness[best_idx]
            n_viable_solutions = np.count_nonzero(self.optimization.constraints_satisfied(ind.tolist())[0] for ind in self.pop)

            log(self.file_name, f"Melhor indivíduo: {best_individual}")
            log(self.file_name, f"Melhor fitness: {best_fitness:.6f}")
            log(self.file_name, f"Contagem de soluções viáves: {n_viable_solutions}")

            print(best_fitness)
            print(n_viable_solutions)


            elapsed_time = time.time() - start_time

        print(self.optimization.constraints_satisfied(best_individual.tolist()))

        return best_individual, best_fitness
