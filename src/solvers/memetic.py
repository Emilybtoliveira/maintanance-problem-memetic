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
    
    def _repair_population(self, pop, resource_window=5, max_time=10) -> np.ndarray:
        # print("Repairing population...")

        # intervention = np.count_nonzero([self.optimization.intervention_constraint_satisfied(individual)[0] for individual in pop])
        # exclusion = np.count_nonzero([self.optimization.exclusion_constraint_satisfied(individual)[0] for individual in pop])
        # resource = np.count_nonzero([self.optimization.resources_constraint_satisfied(individual)[0] for individual in pop])        

        # print(f"Intervention Constraint: {intervention}")
        # print(f"Exclusion Constraint: {exclusion}")
        # print(f"Resource Constraint: {resource}")

        start_time = time.time()

        for index, individual in enumerate(pop):
            if time.time() - start_time > max_time:
                break
            # print(self.optimization.intervention_constraint_satisfied(individual))
            # print(self.optimization.exclusion_constraint_satisfied(individual))
            # print(self.optimization.resources_constraint_satisfied(individual))

            # Verifica se há violações no tempo de inicio das intervenções
            if not self.optimization.intervention_constraint_satisfied(individual)[0]:
                for i, start_time in enumerate(individual):
                    if (start_time > self.problem.interventions[i].tmax or 
                        start_time + self.problem.interventions[i].delta[start_time - 1] > self.problem.time_horizon.time_steps):

                        max_allowable_start = self.problem.interventions[i].tmax
                        for i in range(self.problem.interventions[i].tmax, -1, -1):
                            if i + self.problem.interventions[i].delta[i] <= self.problem.time_horizon.time_steps:
                                max_allowable_start = i
                                break
                        
                        individual[i] = np.random.randint(1, max_allowable_start)
                

            # Verifica se há conflitos de exclusão
            if not self.optimization.exclusion_constraint_satisfied(individual)[0]:
                for idx in range(len(self.problem.exclusions)):
                    if time.time() - start_time > max_time:
                        break
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
                                # else:      
                                #     print("Não foi possível reparar conflito de exclusão entre intervenções", excl_interventions[i], "e", excl_interventions[j])
                

            # Verifica se há conflitos de recursos
            if not self.optimization.resources_constraint_satisfied(individual)[0]:
                _, _, underused, overused = self.optimization.resources_constraint_satisfied(individual)          
                improved = False
                for t in overused.keys():
                    if time.time() - start_time > max_time:
                        break
                    for intervention in overused[t]:
                        # tenta deslocar a intervenção
                        new = self._try_shift(individual, intervention, window=resource_window, overuse=True)
                        if new is not None:
                            individual = new
                            pop[index] = individual
                            improved = True
                    # if improved:
                        # print("Successfully repaired overused resource conflict.")
                
                for t in underused.keys():
                    if time.time() - start_time > max_time:
                        break
                    for intervention in underused[t]:
                        # tenta deslocar a intervenção
                        new = self._try_shift(individual, intervention, window=resource_window, overuse=False)
                        if new is not None:
                            individual = new
                            pop[index] = individual
                            improved = True
                        # if improved:
                            # print("Successfully repaired underused resource conflict.")                    

                # if not improved:
                #     print("Não foi possível reparar conflito de recursos.")
                
            # print(self.optimization.resources_constraint_satisfied(individual))

            # print("-----")

        # intervention = np.count_nonzero([self.optimization.intervention_constraint_satisfied(individual)[0] for individual in pop])
        # exclusion = np.count_nonzero([self.optimization.exclusion_constraint_satisfied(individual)[0] for individual in pop])
        # resource = np.count_nonzero([self.optimization.resources_constraint_satisfied(individual)[0] for individual in pop])        

        # print(f"Intervention Constraint: {intervention}")
        # print(f"Exclusion Constraint: {exclusion}")
        # print(f"Resource Constraint: {resource}")
        return pop


    def _feasible_start(self, individual, i, new_start) -> bool:
        """
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
            1, self.problem.interventions[mutation_idx].tmax + 1 # Mutação garante que o tempo de início seja válido
        )
        return individual
    
    def _evaluate_individual(self, individual) -> float:
        _, penalty = self.optimization.constraints_satisfied(individual)
        fitness, _, _ = self.optimization.objective_function(individual, penalty)
        return fitness
    

    def _local_search(self, individual, max_time = 10, window=10, first_improvement = True) -> np.ndarray:
        """
        Busca local que tenta fazer trocar nos tempos de início das intervenções para melhorar o fitness.
        """
        best = individual.copy()
        best_fitness = self._evaluate_individual(best)

        improved = True
        start = time.time()
        while improved and (time.time() - start < max_time):
            improved = False
            for i in range(len(individual)):
                if time.time() - start >= max_time:
                    break
                current_start = best[i]
                best_local_fitness = best_fitness

                for delta in range(-window, window):
                    if delta == 0:
                        continue
                    new_start = current_start + delta

                    if new_start < 1 or new_start > self.problem.interventions[i].tmax:
                        continue

                    candidate = deepcopy(best)
                    candidate[i] = new_start
                    new_fitness = self._evaluate_individual(candidate)

                    if new_fitness < best_local_fitness:
                        best = candidate
                        best_fitness = new_fitness
                        improved = True
                        if first_improvement:
                            break
            
        # print("Local search completed. Best fitness:", best_fitness)

        return best

    def _generate_new_population(self, repair_children=False) -> np.ndarray:
        new_pop = []

        while len(new_pop) < self.pop_size:
            parent_a, parent_b = self._select_parents(strategy="roulette")
            if np.random.rand() > self.crossover_rate:
                child = self._crossover(parent_a, parent_b, strategy="uniform") 
                
                child = self._local_search(child)

                if np.random.rand() < self.mutation_rate:
                    child = self._mutate(child)

                # child = self._local_search(child)
                new_pop.append(child)
        
        if repair_children:     
            new_pop = self._repair_population(np.array(new_pop))  

        return new_pop
    
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

        self.pop = self._init_populaton(self.pop_size)
        self.pop = self._repair_population(self.pop)

        self.fitness = np.array(
            [self._evaluate_individual(ind) for ind in self.pop]
        )

        elapsed_time = time.time() - start_time
        while elapsed_time < self.time_limit:
            log(f"{self.file_name}", f"Elapsed time: {elapsed_time:.2f} seconds.")                
            
            new_pop = self._generate_new_population(repair_children=False)
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
            n_viable_solutions = np.count_nonzero([self.optimization.constraints_satisfied(ind)[0] for ind in self.pop])

            log(self.file_name, f"Melhor indivíduo: {best_individual}")
            log(self.file_name, f"Melhor fitness: {best_fitness:.6f}")
            log(self.file_name, f"Contagem de soluções viáves: {n_viable_solutions}")

            print(best_fitness)
            print(n_viable_solutions)

            elapsed_time = time.time() - start_time

        print(self.optimization.constraints_satisfied(best_individual))

        return best_individual, best_fitness