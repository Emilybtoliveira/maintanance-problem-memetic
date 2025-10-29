from pydantic import BaseModel


class Intervention(BaseModel):
    """
    Represents an intervention in the model.

    Attributes:
    - name (str): Name of the intervention.
    - tmax (int): A pre-computed value corresponding to the latest possible starting time for a given intervention.
    - delta (list[int]): List of integers representing the duration of the intervention.
    - resource_workload (list[Resource]): List of resources required for the intervention.
    """

    name: str = ""
    tmax: int = 0
    delta: list[int] = []
    resource_workload: dict[str, dict[str, dict[str, float]]] = {}
    risk: dict[str, dict[str, list[float]]] = {}
