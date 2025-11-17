import numpy as np
from numba import njit
from datamodels.problem import Problem


class Optimization:
    """
    Class responsible for building the objective function and checking the constraints.
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self._prepare_data_structures()

    def _prepare_data_structures(self):
        """
        Prepare all data structures for use with Numba
        """
        T = self.problem.time_horizon.time_steps
        n_interventions = len(self.problem.interventions)
        n_resources = len(self.problem.resources)
        max_scenarios = max(self.problem.scenarios)

        # Arrays for risk data
        self.risk_array = np.zeros(
            (T + 1, n_interventions, T + 1, max_scenarios), dtype=np.float32
        )
        for i, intervention in enumerate(self.problem.interventions):
            for t in range(1, T + 1):
                for start_time in range(1, T + 1):
                    try:
                        risks = intervention.risk[str(t)][str(start_time)]
                        for s, risk in enumerate(risks):
                            self.risk_array[t, i, start_time, s] = risk
                    except KeyError:
                        continue

        # Arrays for intervention durations
        self.deltas = np.zeros((n_interventions, T + 1), dtype=np.int32)
        for i, intervention in enumerate(self.problem.interventions):
            for t in range(len(intervention.delta)):
                self.deltas[i, t] = intervention.delta[t]

        # Arrays for resource limits
        self.resource_mins = np.zeros((n_resources, T))
        self.resource_maxs = np.zeros((n_resources, T))
        for i, resource in enumerate(self.problem.resources):
            self.resource_mins[i] = resource.min
            self.resource_maxs[i] = resource.max

        # Arrays for resource workload
        self.resource_workload = np.zeros(
            (n_resources, n_interventions, T + 1, T + 1), dtype=np.float32
        )
        for r, resource in enumerate(self.problem.resources):
            for i, intervention in enumerate(self.problem.interventions):
                try:
                    workload = intervention.resource_workload[resource.name]
                    for t in range(1, T + 1):
                        for start_time in range(1, T + 1):
                            try:
                                self.resource_workload[r, i, t, start_time] = workload[
                                    str(t)
                                ][str(start_time)]
                            except KeyError:
                                continue
                except KeyError:
                    continue

        # Arrays for exclusions
        n_exclusions = len(self.problem.exclusions)
        self.exclusion_pairs = np.zeros((n_exclusions, 2), dtype=np.int32)
        self.exclusion_seasons = np.zeros((n_exclusions, T + 1), dtype=np.int32)

        for idx, excl in enumerate(self.problem.exclusions):
            i1 = next(
                i
                for i, inv in enumerate(self.problem.interventions)
                if inv.name == excl.interventions[0]
            )
            i2 = next(
                i
                for i, inv in enumerate(self.problem.interventions)
                if inv.name == excl.interventions[1]
            )
            self.exclusion_pairs[idx] = [i1, i2]
            for t in excl.season.duration:
                self.exclusion_seasons[idx, t] = 1

        # Other important parameters
        self.tmax_values = np.array(
            [intervention.tmax for intervention in self.problem.interventions],
            dtype=np.int32,
        )
        self.time_steps = T
        self.scenarios_array = np.array(self.problem.scenarios)

    def _objective_function(self, start_times, penalty=0) -> float:
        """
        Optimized objective function interface
        """
        return _numba_objective_function(
            np.array(start_times),
            self.risk_array,
            self.deltas,
            self.time_steps,
            self.problem.quantile,
            self.problem.alpha,
            self.scenarios_array,
            penalty,
        )

    def _intervention_constraint_satisfied(self, start_times) -> tuple[bool, float]:
        return _numba_intervention_constraint(
            np.array(start_times), self.deltas, self.tmax_values, self.time_steps
        )

    def _resources_constraint_satisfied(self, start_times) -> tuple[bool, float]:
        return _numba_resources_constraint(
            np.array(start_times),
            self.resource_workload,
            self.resource_mins,
            self.resource_maxs,
            self.deltas,
            self.time_steps,
        )

    def _exclusion_constraint_satisfied(self, start_times) -> tuple[bool, float]:
        return _numba_exclusion_constraint(
            np.array(start_times),
            self.deltas,
            self.exclusion_pairs,
            self.exclusion_seasons,
        )

    def _constraints_satisfied(self, start_times) -> tuple[bool, float]:
        start_times_array = np.array(start_times, dtype=np.int32)
        penalty = 0.0

        violated, pen = _numba_intervention_constraint(
            start_times_array, self.deltas, self.tmax_values, self.time_steps
        )
        if violated:
            penalty += pen * 1e6

        violated, pen = _numba_resources_constraint(
            start_times_array,
            self.resource_workload,
            self.resource_mins,
            self.resource_maxs,
            self.deltas,
            self.time_steps,
        )
        if violated:
            penalty += pen * 1e6

        violated, pen = _numba_exclusion_constraint(
            start_times_array, self.deltas, self.exclusion_pairs, self.exclusion_seasons
        )
        if violated:
            penalty += pen * 1e6

        return penalty == 0, penalty


@njit(fastmath=True)
def _numba_objective_function(
    start_times, risk_array, deltas, T, quantile, alpha, scenarios, penalty
):
    mean_risk = 0.0
    expected_excess = 0.0

    for t in range(1, T + 1):
        risk_t = 0.0
        risk_by_scenario = np.zeros(scenarios[t - 1])

        for i in range(len(start_times)):
            start_time = int(start_times[i])
            if start_time <= t < start_time + deltas[i, start_time - 1]:
                for s in range(scenarios[t - 1]):
                    risk_value = risk_array[t, i, start_time, s]
                    risk_t += risk_value
                    risk_by_scenario[s] += risk_value

        risk_t /= max(1, scenarios[t - 1])
        mean_risk += risk_t

        risk_by_scenario = np.sort(risk_by_scenario)
        quantile_index = int(np.ceil(quantile * len(risk_by_scenario))) - 1
        excess_t = 0.0
        if len(risk_by_scenario) > 0:
            excess_t = max(0.0, risk_by_scenario[quantile_index] - risk_t)

        expected_excess += excess_t

    mean_risk /= T
    expected_excess /= T
    objective = (alpha * mean_risk) + ((1.0 - alpha) * expected_excess)

    return objective + penalty, mean_risk, expected_excess


@njit
def _numba_intervention_constraint(start_times, deltas, tmax_values, time_horizon):
    penalty = 0.0
    for i in range(len(start_times)):
        start_time = int(start_times[i])

        if start_time < 0 or start_time > tmax_values[i]:
            penalty += float(start_time - tmax_values[i])

        if start_time > 0 and start_time <= len(deltas[i]):
            duration = deltas[i, start_time - 1]
            end_time = start_time + duration - 1

            if end_time > time_horizon:
                penalty += float(end_time - time_horizon)

    return penalty > 0, penalty


@njit
def _numba_resources_constraint(
    start_times, resource_workload, resource_mins, resource_maxs, deltas, T
):
    eps = 1e-6
    penalty = 0.0

    for t in range(1, T + 1):
        for r in range(len(resource_mins)):
            total_resource_usage = 0.0

            for i in range(len(start_times)):
                start_time = int(start_times[i])
                if start_time <= t <= start_time + deltas[i, start_time - 1] - 1:
                    total_resource_usage += resource_workload[r, i, t, start_time]

            if total_resource_usage < resource_mins[r, t - 1] - eps:
                penalty += resource_mins[r, t - 1] - total_resource_usage
            elif total_resource_usage > resource_maxs[r, t - 1] + eps:
                penalty += total_resource_usage - resource_maxs[r, t - 1]

    return penalty > 0, penalty


@njit
def _numba_exclusion_constraint(
    start_times, deltas, exclusion_pairs, exclusion_seasons
):
    penalty = 0.0

    for idx in range(len(exclusion_pairs)):
        i1, i2 = exclusion_pairs[idx]
        start1 = int(start_times[i1])
        end1 = start1 + deltas[i1, start1 - 1] - 1

        start2 = int(start_times[i2])
        end2 = start2 + deltas[i2, start2 - 1] - 1

        t_start = max(start1, start2)
        t_end = min(end1, end2)

        if t_start <= t_end:
            for t in range(t_start, t_end + 1):
                if exclusion_seasons[idx, t] == 1:
                    penalty += 1

    return penalty > 0, penalty
