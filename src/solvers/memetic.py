from copy import deepcopy
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
        crossover_rate: float = 0.75,
        mutation_rate: float = 0.15,
        time_limit: int = 60 * 5,
        tol: float = 1e-6,
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

    def _init_population(self, pop_size) -> np.ndarray:
        pop = []

        for _ in range(pop_size):
            individual = []
            for i in range(len(self.problem.interventions)):
                intervention = self.problem.interventions[i]
                max_valid_start = intervention.tmax

                for t in range(intervention.tmax, 0, -1):
                    if (
                        t + intervention.delta[t - 1]
                        <= self.problem.time_horizon.time_steps
                    ):
                        max_valid_start = t
                        break

                start_time = np.random.randint(1, max(2, max_valid_start + 1))
                individual.append(start_time)

            pop.append(individual)

        return np.array(pop)

    def _repair_individual(self, individual) -> np.ndarray:
        for i in range(len(individual)):
            intervention = self.problem.interventions[i]

            if individual[i] < 1 or individual[i] > intervention.tmax:
                individual[i] = np.random.randint(1, intervention.tmax + 1)

            delta = intervention.delta[individual[i] - 1]
            if individual[i] + delta > self.problem.time_horizon.time_steps:
                max_start = max(1, self.problem.time_horizon.time_steps - delta)
                individual[i] = min(max_start, intervention.tmax)

        return individual

    def _select_parents(self):
        def tournament_selection(k=3):
            contenders = np.random.choice(range(self.pop_size), k, replace=False)
            return min(contenders, key=lambda ind: self.fitness[ind])

        p1 = tournament_selection()
        p2 = tournament_selection()
        return self.pop[p1], self.pop[p2]

    def _crossover(self, parent_a, parent_b):
        n = len(parent_a)
        mask = np.random.rand(n) < 0.5
        child = np.where(mask, parent_a, parent_b)
        return child

    def _mutate(self, individual):
        strategy = np.random.choice(["single", "multiple", "shift"])

        if strategy == "single":
            idx = np.random.randint(0, len(individual))
            individual[idx] = np.random.randint(
                1, self.problem.interventions[idx].tmax + 1
            )

        elif strategy == "multiple":
            n_mutations = np.random.randint(2, min(4, len(individual) + 1))
            indices = np.random.choice(len(individual), n_mutations, replace=False)
            for idx in indices:
                individual[idx] = np.random.randint(
                    1, self.problem.interventions[idx].tmax + 1
                )

        else:
            offset = np.random.randint(-3, 4)
            for i in range(len(individual)):
                new_val = individual[i] + offset
                new_val = np.clip(new_val, 1, self.problem.interventions[i].tmax)
                individual[i] = new_val

        return individual

    def _evaluate_individual(self, individual) -> float:
        _, penalty = self.optimization.constraints_satisfied(individual)
        fitness, _, _ = self.optimization.objective_function(individual, penalty)
        return fitness

    def _shift_move(self, individual, intervention_idx, new_time):
        candidate = individual.copy()
        candidate[intervention_idx] = new_time
        return candidate

    def _swap_move(self, individual, i1, i2):
        candidate = individual.copy()
        candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        return candidate

    def _is_feasible_move(self, candidate):
        interv_ok = self.optimization.intervention_constraint_satisfied(candidate)[0]
        excl_ok = self.optimization.exclusion_constraint_satisfied(candidate)[0]
        return interv_ok and excl_ok

    def _local_search(
        self, individual, max_time=3, max_non_improving=100
    ) -> np.ndarray:
        best = individual.copy()
        best_fitness = self._evaluate_individual(best)
        current = best.copy()
        current_fitness = best_fitness

        start = time.time()
        non_improving_moves = 0
        iteration = 0
        use_shift = True

        while time.time() - start < max_time:
            iteration += 1
            improved = False

            if use_shift:
                candidates = []
                for i in range(len(current)):
                    current_start = current[i]
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_start = current_start + delta
                        if 1 <= new_start <= self.problem.interventions[i].tmax:
                            candidate = self._shift_move(current, i, new_start)
                            if self._is_feasible_move(candidate):
                                candidates.append(candidate)

                if candidates:
                    candidate = candidates[np.random.randint(len(candidates))]
                    candidate_fitness = self._evaluate_individual(candidate)

                    if candidate_fitness < current_fitness or (iteration % 1000 == 0):
                        current = candidate
                        current_fitness = candidate_fitness

                        if candidate_fitness < best_fitness:
                            best = candidate
                            best_fitness = candidate_fitness
                            improved = True
                            non_improving_moves = 0
            else:
                candidates = []
                indices = list(range(len(current)))
                np.random.shuffle(indices)

                for idx in range(min(10, len(indices))):
                    i1 = indices[idx]
                    i2 = indices[(idx + 1) % len(indices)]
                    candidate = self._swap_move(current, i1, i2)
                    if self._is_feasible_move(candidate):
                        candidates.append(candidate)

                if candidates:
                    candidate = candidates[np.random.randint(len(candidates))]
                    candidate_fitness = self._evaluate_individual(candidate)

                    if candidate_fitness < current_fitness or (iteration % 1000 == 0):
                        current = candidate
                        current_fitness = candidate_fitness

                        if candidate_fitness < best_fitness:
                            best = candidate
                            best_fitness = candidate_fitness
                            improved = True
                            non_improving_moves = 0

            if not improved:
                non_improving_moves += 1
                if non_improving_moves >= max_non_improving:
                    use_shift = not use_shift
                    non_improving_moves = 0

        return best

    def _generate_new_population(self) -> np.ndarray:
        new_pop = []

        while len(new_pop) < self.pop_size:
            parent_a, parent_b = self._select_parents()

            if np.random.rand() < self.crossover_rate:
                child = self._crossover(parent_a, parent_b)
            else:
                child = parent_a.copy() if np.random.rand() < 0.5 else parent_b.copy()

            if np.random.rand() < self.mutation_rate:
                child = self._mutate(child)

            child = self._repair_individual(child)
            new_pop.append(child)

        new_pop = np.array(new_pop)
        fitness_new = np.array([self._evaluate_individual(ind) for ind in new_pop])
        n_to_improve = max(1, int(0.2 * self.pop_size))
        best_indices = np.argsort(fitness_new)[:n_to_improve]

        for idx in best_indices:
            new_pop[idx] = self._local_search(new_pop[idx])

        return new_pop

    def _update_population(self, pop, new_pop):
        best_idx = np.argmin(self.fitness)
        best_individual = pop[best_idx]
        new_pop[0] = best_individual
        return new_pop

    def _restart_population(self, pop):
        percentage_to_keep = 0.2
        n_to_keep = int(self.pop_size * percentage_to_keep)

        best_idxs = np.argsort(self.fitness)[:n_to_keep]
        best_individuals = pop[best_idxs]
        new_pop = self._init_population(self.pop_size - n_to_keep)
        return np.vstack((new_pop, best_individuals))

    def _has_converged(self) -> bool:
        return np.all(np.abs(self.fitness - self.fitness.mean()) < self.tol)

    def optimize(self) -> tuple[np.ndarray, float]:
        start_time = time.time()
        generation = 0

        self.pop = self._init_population(self.pop_size)
        for i in range(self.pop_size):
            self.pop[i] = self._repair_individual(self.pop[i])

        self.fitness = np.array([self._evaluate_individual(ind) for ind in self.pop])

        best_fitness_history = []
        no_improvement_count = 0

        while time.time() - start_time < self.time_limit:
            generation += 1
            elapsed = time.time() - start_time

            log(f"{self.file_name}", f"Gen {generation} | Time: {elapsed:.1f}s")

            new_pop = self._generate_new_population()
            self.pop = self._update_population(self.pop, new_pop)
            self.fitness = np.array(
                [self._evaluate_individual(ind) for ind in self.pop]
            )

            best_idx = np.argmin(self.fitness)
            best_fitness = self.fitness[best_idx]
            best_fitness_history.append(best_fitness)

            if len(best_fitness_history) > 20:
                recent_improvement = best_fitness_history[-20] - best_fitness
                if recent_improvement < 1e-3:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

            if no_improvement_count > 5 or (generation > 10 and self._has_converged()):
                log(self.file_name, "Reiniciando população por estagnação")
                self.pop = self._restart_population(self.pop)
                self.fitness = np.array(
                    [self._evaluate_individual(ind) for ind in self.pop]
                )
                no_improvement_count = 0

            n_viable = np.sum(
                [self.optimization.constraints_satisfied(ind)[0] for ind in self.pop]
            )
            log(
                self.file_name,
                f"Best: {best_fitness} | Viable: {n_viable}/{self.pop_size}",
            )

            print(f"Gen {generation}: {best_fitness} | Viable: {n_viable}")

        best_idx = np.argmin(self.fitness)
        best_individual = self.pop[best_idx]

        print(self.optimization.constraints_satisfied(best_individual))

        return best_individual, self.fitness[best_idx]
