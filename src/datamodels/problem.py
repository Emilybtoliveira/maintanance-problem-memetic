from pydantic import BaseModel
from .exclusion import Exclusion
from .resource import Resource
from .intervention import Intervention
from .time_horizon import TimeHorizon


class Problem(BaseModel):
    """
    Represents a maintenance planning problem.

    Attributes:
        resources (list[Resource]): A list of resources available for maintenance.
        interventions (list[Intervention]): A list of interventions to be performed.
        exclusions (list[Exclusion]): A list of exclusions between interventions.
        time_horizon (TimeHorizon): The time horizon for the maintenance planning.
        scenarios (list[int]): A list of scenarios for stochastic optimization.
        quantile (float): The quantile value for stochastic optimization.
        alpha (float): The alpha value for stochastic optimization.
        computation_time (float): The computation time for solving the problem.
    """

    resources: list[Resource] = []
    interventions: list[Intervention] = []
    exclusions: list[Exclusion] = []
    time_horizon: TimeHorizon
    scenarios: list[int] = []
    quantile: float | None = None
    alpha: float | None = None
    computation_time: float | None = None
