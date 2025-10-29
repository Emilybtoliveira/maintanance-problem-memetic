from pydantic import BaseModel
from . import Season


class Exclusion(BaseModel):
    """
    Represents an exclusion in the maintenance planning problem.

    Attributes:
        name (str): The name of the exclusion.
        interventions (list[Intervention]): A list of interventions associated with the exclusion.
        season (str): The season during which the exclusion is applicable.
    """

    name: str = ""
    interventions: list[str] = []
    season: Season
