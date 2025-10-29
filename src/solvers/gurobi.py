import gurobipy as gp
from gurobipy import GRB


class Gurobi:
    def __init__(self, time_limit, problem):
        self.time_limit = time_limit
        self.problem = problem
        self.model = gp.Model()
        self.model.setParam("TimeLimit", time_limit)

    def optimize(self):
        self._create_variables()
        self._objective_function()
        self._intervention_constraint()
        self._resource_constraint()
        self._exclusion_constraint()

        self.model.optimize()

        return self.get_solution()

    def _create_variables(self):
        self.x = {
            i: {
                t: self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
                for t in range(1, self.problem.time_horizon.time_steps + 1)
            }
            for i in range(len(self.problem.interventions))
        }

    def _objective_function(self):
        expr = gp.LinExpr()

        for t in range(1, self.problem.time_horizon.time_steps + 1):
            for s in range(self.problem.scenarios[t - 1]):
                for i in range(len(self.problem.interventions)):
                    for st in range(1, self.problem.interventions[i].tmax + 1):
                        try:
                            expr.addTerms(
                                self.problem.interventions[i].risk[str(t)][str(st)][s]
                                / (
                                    self.problem.time_horizon.time_steps
                                    * self.problem.scenarios[t - 1]
                                ),
                                self.x[i][st],
                            )
                        except KeyError:
                            pass

        self.model.setObjective(
            expr,
            GRB.MINIMIZE,
        )

    def _intervention_constraint(self):
        for i in range(len(self.problem.interventions)):
            self.model.addConstr(
                gp.quicksum(
                    self.x[i][t]
                    for t in range(1, self.problem.interventions[i].tmax + 1)
                )
                == 1
            )

    def _resource_constraint(self):
        for r in range(len(self.problem.resources)):
            for t in range(1, self.problem.time_horizon.time_steps + 1):
                expr = gp.LinExpr()

                for i in range(len(self.problem.interventions)):
                    for st in range(1, self.problem.interventions[i].tmax + 1):
                        try:
                            expr.addTerms(
                                self.problem.interventions[i].resource_workload[
                                    self.problem.resources[r].name
                                ][str(t)][str(st)],
                                self.x[i][st],
                            )
                        except KeyError:
                            pass

                self.model.addConstr(self.problem.resources[r].min[t - 1] <= expr)
                self.model.addConstr(expr <= self.problem.resources[r].max[t - 1])

    def _exclusion_constraint(self):
        for e in self.problem.exclusions:
            i1, i2, season = (
                e.interventions[0],
                e.interventions[1],
                e.season,
            )

            i1 = next(
                i
                for i, intervention in enumerate(self.problem.interventions)
                if intervention.name == i1
            )
            i2 = next(
                i
                for i, intervention in enumerate(self.problem.interventions)
                if intervention.name == i2
            )

            for t in season.duration:
                expr = gp.LinExpr()

                for st in range(1, self.problem.interventions[i1].tmax + 1):
                    if st <= t <= st + self.problem.interventions[i1].delta[st - 1] - 1:
                        expr.add(self.x[i1][st])
                for st in range(1, self.problem.interventions[i2].tmax + 1):
                    if st <= t <= st + self.problem.interventions[i2].delta[st - 1] - 1:
                        expr.add(self.x[i2][st])

                self.model.addConstr(expr <= 1)

    def get_objective_value(self):
        return self.model.objVal

    def get_solution(self):
        gurobi_solution = []

        for i in range(len(self.problem.interventions)):
            for t in range(1, self.problem.time_horizon.time_steps + 1):
                if self.x[i][t].x > 0.5:
                    gurobi_solution.append(t)

        return gurobi_solution
