import time
import numpy as np

from datamodels.problem import Problem
from optimization import Optimization
from utils.log import log


class GeneticAlgorithm:
    def __init__(
        self,
        file_name: str,
        problem: Problem,
        optimization: Optimization,
        pop: np.ndarray,
        pop_size: int,
        fitness: np.ndarray,
        obj_func: callable,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        time_limit: int = 60 * 5,  # seconds
        tol: int = 1e-6,
    ):
        self.file_name = file_name
        self.problem = problem
        self.optimization = optimization
        self.pop = pop
        self.pop_size = pop_size
        self.fitness = fitness
        self.obj_func = obj_func
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.time_limit = time_limit
        self.tol = tol

    def _reproduce(self, a, b):
        """
        Reproduce two individuals.
        """
        n = len(a)
        c = np.random.randint(1, n)
        return np.concatenate((a[:c], b[c:]), axis=0)

    def _evaluate_individual(self, i):
        """
        Evaluate and generate a new individual.
        """
        idx = np.random.choice(self.pop_size, 2, replace=False)
        a, b = self.pop[idx]

        if np.random.rand() < self.crossover_rate:
            child = self._reproduce(a, b)
        else:
            child = a

        # Mutation
        if np.random.rand() < self.mutation_rate:
            mutation_idx = np.random.randint(0, len(child))
            child[mutation_idx] = np.random.randint(
                1, self.problem.time_horizon.time_steps
            )

        _, child_penalty = self.optimization._constraints_satisfied(child.tolist())
        child_fitness = self.obj_func(child, child_penalty)[0]

        # print(f"GA - Fitness child: {child_fitness}")

        if child_fitness < self.fitness[i]:
            return child, child_fitness
        else:
            return self.pop[i], self.fitness[i]

    def optimize(self):
        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time
            # log(f"{self.file_name}", f"GA - Elapsed time: {elapsed_time:.2f} seconds.")
            if elapsed_time > self.time_limit:
                log(
                    f"{self.file_name}",
                    f"GA - Maximum execution time reached: {elapsed_time:.2f} seconds.",
                )
                return self.pop, self.fitness

            new_pop = np.zeros_like(self.pop)

            results = [self._evaluate_individual(i) for i in range(self.pop_size)]

            for i, (new_individual, new_fitness) in enumerate(results):
                new_pop[i] = new_individual
                self.fitness[i] = new_fitness

            if np.all(np.abs(self.fitness - self.fitness.mean()) < self.tol):
                log(f"{self.file_name}", "GA - Converged")
                log(
                    f"{self.file_name}",
                    f"GA - Solution: {self.pop[np.argmin(self.fitness)]}",
                )
                log(f"{self.file_name}", f"GA - Fitness: {np.min(self.fitness)}")
                return self.pop, self.fitness

            self.pop = new_pop
