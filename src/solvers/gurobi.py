import gurobipy as gp
from gurobipy import GRB


class Gurobi:
    def __init__(self, time_limit, problem, threads=1):
        self.time_limit = time_limit
        self.problem = problem
        self.threads = threads
        self.model = gp.Model()

        self.model.setParam("TimeLimit", self.time_limit)
        self.model.setParam("Threads", self.threads)
        # self.model.setParam("OutputFlag", 0)  # Suppress output

        self.model.Params.MIPFocus = 1  # 1=feasibility, 2=optimality, 3=bound
        self.model.Params.PoolSearchMode = 2  # 2=extensive search for diverse solutions
        self.model.Params.PoolSolutions = 30  # Keep up to 30 solutions
        self.model.Params.PoolGap = 1.0  # Accept solutions within 100% of best

    def optimize(self):
        self._create_variables()
        self._objective_function()
        self._intervention_constraint()
        self._resource_constraint()
        self._exclusion_constraint()

        self.model.optimize()

        return self.get_pool_solutions()

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

        # Zero objective function to focus on feasibility
        # self.model.setObjective(0, GRB.MINIMIZE)

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

    def get_pool_solutions(self):
        pool_solutions = []
        n_solutions = self.model.SolCount

        for s in range(n_solutions):
            self.model.Params.SolutionNumber = s

            gurobi_solution = []

            for i in range(len(self.problem.interventions)):
                start_time = None
                for t in range(1, self.problem.time_horizon.time_steps + 1):
                    if self.x[i][t].Xn > 0.5:
                        start_time = t
                        break
                if start_time is None:
                    start_time = 0
                gurobi_solution.append(start_time)

            pool_solutions.append(gurobi_solution)

        return pool_solutions

    def get_solution(self):
        gurobi_solution = []

        for i in range(len(self.problem.interventions)):
            for t in range(1, self.problem.time_horizon.time_steps + 1):
                if self.x[i][t].x > 0.5:
                    gurobi_solution.append(t)

        return gurobi_solution
