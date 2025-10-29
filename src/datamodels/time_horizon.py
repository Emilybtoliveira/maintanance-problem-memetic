from pydantic import BaseModel


class TimeHorizon(BaseModel):
    """
    Represents the time horizon of a maintenance planning problem.

    Attributes:
        time_steps (int): The number of time steps in the time horizon.
    """

    time_steps: int
