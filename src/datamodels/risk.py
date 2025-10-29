from pydantic import BaseModel


class Risk(BaseModel):
    """
    Represents a risk in the maintenance planning problem.

    Attributes:
    - time_period (int): The time period in which the risk occurs.
    - scenario (list[int]): A list of scenarios associated with the risk.
    """

    time_step: int
    start_time_step: list[int]
    scenarios: list[int]
