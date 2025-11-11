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
        gb_solution: np.ndarray,
        pop_size: int,
        crossover_rate: float = 0.75,
        mutation_rate: float = 0.15,
        time_limit: int = 60 * 5,
        remaining_time: int = 60 * 5,
        tol: float = 1e-6,
    ):
        self.file_name = file_name
        self.problem = problem
        self.gb_solution = gb_solution
        self.pop_size = pop_size
        self.fitness = POSITIVE_INFINITY * np.ones(pop_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.time_limit = time_limit
        self.remaining_time = remaining_time
        self.tol = tol
        self.optimization = Optimization(problem)
        self.pop = None
    # ==========================
    # Memetic Algorithm Functions
    # ==========================

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
    
    def _repair_population(self, pop) -> np.ndarray:
        """
        Função de reparo da população como um todo. Não utilizada, porque se mostrou
        muito custosa.
        """
        for index, individual in enumerate(pop):
            # Verifica se há violações no tempo de inicio das intervenções
            for j in range(len(individual)):
                intervention = self.problem.interventions[j]

                if individual[j] < 1 or individual[j] > intervention.tmax:
                    individual[j] = np.random.randint(1, intervention.tmax + 1)

                delta = intervention.delta[individual[j] - 1]
                if individual[j] + delta > self.problem.time_horizon.time_steps:
                    max_start = max(1, self.problem.time_horizon.time_steps - delta)
                    individual[j] = min(max_start, intervention.tmax)
                
                pop[index] = individual                

            # Verifica se há conflitos de exclusão
            if not self.optimization.exclusion_constraint_satisfied(individual)[0]:
                for idx in range(len(self.problem.exclusions)):
                    excl_interventions_names = self.problem.exclusions[idx].interventions
                    excl_interventions = [
                        next(
                            i
                            for i, inv in enumerate(self.problem.interventions)                            
                            if inv.name == inv_name
                        ) 
                        for inv_name in excl_interventions_names
                    ]

                    start_times = [individual[i] for i in excl_interventions]
                    deltas = [self.problem.interventions[i].delta[individual[i] - 1] for i in excl_interventions]
                    end_times = [start_times[i] + deltas[i] - 1 for i, _ in enumerate(excl_interventions)]

                    for i in range(len(excl_interventions)):
                        for j in range(i + 1, len(excl_interventions)):
                            # Verifica se há sobreposição
                            if not (end_times[i] < start_times[j] or end_times[j] < start_times[i]):
                                # Ajusta o tempo de início de uma das intervenções se for viável
                                if end_times[i] + 1 <= self.problem.interventions[excl_interventions[i]].tmax:
                                    individual[excl_interventions[j]] = end_times[i] + 1
                                    start_times[j] = individual[excl_interventions[j]]
                                    end_times[j] = start_times[j] + deltas[j] - 1  
                                elif start_times[i] - deltas[j] >= 1:
                                    individual[excl_interventions[j]] = start_times[i] - deltas[j]
                                    start_times[j] = individual[excl_interventions[j]]
                                    end_times[j] = start_times[j] + deltas[j] - 1
                                elif end_times[j] + 1 <= self.problem.interventions[excl_interventions[j]].tmax: 
                                    individual[excl_interventions[i]] = end_times[j] + 1
                                    start_times[i] = individual[excl_interventions[i]]
                                    end_times[i] = start_times[i] + deltas[i] - 1
                                elif start_times[j] - deltas[i] >= 1:
                                    individual[excl_interventions[i]] = start_times[j] - deltas[i]
                                    start_times[i] = individual[excl_interventions[i]]
                                    end_times[i] = start_times[i] + deltas[i] - 1
    
                                pop[index] = individual


            # Verifica se há conflitos de recursos
            if not self.optimization.resources_constraint_satisfied(individual)[0]:
                _, _, underused, overused = self.optimization.resources_constraint_satisfied(individual)          
                for t in overused.keys():
                    for intervention in overused[t]:
                        # tenta deslocar a intervenção
                        new = self._try_shift(individual, intervention, overuse=True)
                        if new is not None:
                            individual = new
                            pop[index] = individual
                
                for t in underused.keys():
                    for intervention in underused[t]:
                        # tenta deslocar a intervenção
                        new = self._try_shift(individual, intervention, overuse=False)
                        if new is not None:
                            individual = new
                            pop[index] = individual
        return pop

    def _feasible_start(self, individual, i, new_start) -> bool:
        """
        Função auxiliar de _repair_population.
        Testa se ao colocar a intervenção i em new_start as restrições básicas
        (intervention + exclusion) permanecem válidas. Não checa recursos aqui.
        Usa os validadores da Optimization para segurança. 
        """
        ind_copy = deepcopy(individual)
        ind_copy[i] = new_start
        interv_ok = self.optimization.intervention_constraint_satisfied(ind_copy)[0]
        excl_ok = self.optimization.exclusion_constraint_satisfied(ind_copy)[0]
        return interv_ok and excl_ok

    def _try_shift(self, individual, i, window=5, overuse=True) -> np.ndarray:
        """
        Função auxiliar de _repair_population.
        Tenta mover a intervenção i para aliviar sobrecarga/subcarga de recursos.
        Busca deslocamentos em [-window .. +window]. Retorna novo start
        se encontrou um deslocamento que melhora (ou mantém) e é factível.
        Critério de melhoria: reduz o número de tempos com overuse (global na solução).
        """
        current_start = int(individual[i])
        def count(ind):
            _, _, underused, overused = self.optimization.resources_constraint_satisfied(ind)
            if overuse:
                return sum(1 for t in overused if len(overused[t]) > 0)
            else:
                return sum(1 for t in underused if len(underused[t]) > 0)

        base_count = count(individual)

        # tenta deslocamentos 1, -1, 2, -2, ...
        shifts = []
        for d in range(1, window + 1):
            shifts.append(d)
            shifts.append(-d)

        for d in shifts:
            new_start = current_start + d            
            if not self._feasible_start(individual, i, new_start):
                continue

            ind_copy = deepcopy(individual)
            ind_copy[i] = new_start
            new_over_count = count(ind_copy)

            # se reduziu o número de tempos sobrecarregados/subcarregados, aceita 
            if new_over_count < base_count:
                return ind_copy

        return None


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
        """
        Seleção de pais via torneio.
        """
        def tournament_selection(k=3):
            contenders = np.random.choice(range(self.pop_size), k, replace=False)
            return min(contenders, key=lambda ind: self.fitness[ind])

        p1 = tournament_selection()
        p2 = tournament_selection()
        return self.pop[p1], self.pop[p2]

    def _crossover(self, parent_a, parent_b):
        """
        Crossover uniforme.
        """
        n = len(parent_a)
        mask = np.random.rand(n) < 0.5
        child = np.where(mask, parent_a, parent_b)
        return child

    def _mutate(self, individual):
        """
        Mutação com três estratégias: mutação de um gene, múltiplos genes ou
        deslocamento de genes.
        """
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
        """
        Função auxiliar de _local_search.
        Gera um movimento de deslocamento para a intervenção dada.
        """
        candidate = individual.copy()
        candidate[intervention_idx] = new_time
        return candidate

    def _swap_move(self, individual, i1, i2):
        """
        Função auxiliar de _local_search.
        Gera um movimento de troca entre as intervenções i1 e i2.
        """
        candidate = individual.copy()
        candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        return candidate

    def _is_feasible_move(self, candidate):
        """
        Função auxiliar de _local_search.
        Verifica se o movimento gerado é factível.
        """
        interv_ok = self.optimization.intervention_constraint_satisfied(candidate)[0]
        excl_ok = self.optimization.exclusion_constraint_satisfied(candidate)[0]
        return interv_ok and excl_ok

    def _local_search(
        self, individual, max_time=3, max_non_improving=100
    ) -> np.ndarray:
        """
        Busca local híbrida com movimentos de deslocamento e troca.
        Alterna entre os dois tipos de movimento quando um número máximo
        de movimentos sem melhoria é atingido.
        """
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
        """
        Atualização da população via elitismo.
        """
        best_idx = np.argmin(self.fitness)
        best_individual = pop[best_idx]
        new_pop[0] = best_individual
        return new_pop

    def _restart_population(self, pop):
        """
        Reinício da população mantendo os melhores indivíduos."""
        percentage_to_keep = 0.2
        n_to_keep = int(self.pop_size * percentage_to_keep)

        best_idxs = np.argsort(self.fitness)[:n_to_keep]
        best_individuals = pop[best_idxs]
        new_pop = self._init_population(self.pop_size - n_to_keep)
        return np.vstack((new_pop, best_individuals))

    def _has_converged(self) -> bool:
        """
        Verifica se a população convergiu com base na tolerância definida.
        """
        return np.all(np.abs(self.fitness - self.fitness.mean()) < self.tol)

    def optimize(self) -> tuple[np.ndarray, float]:
        start_time = time.time()
        generation = 0

        self.pop = self._init_population(self.pop_size - 1)
        self.pop = np.vstack((self.pop, self.gb_solution))

        for i in range(self.pop_size):
            self.pop[i] = self._repair_individual(self.pop[i])

        self.fitness = np.array([self._evaluate_individual(ind) for ind in self.pop])

        best_fitness_history = []
        no_improvement_count = 0

        while time.time() - start_time < self.remaining_time:
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