from pydantic import BaseModel


class Season(BaseModel):
    """
    Represents a season in the maintenance planning problem.

    Attributes:
        name (str): The name of the season.
            Examples: "Full", "Winter", "Summer", "Inter Season".
        duration (list[int]): The duration of the season in months.
    """

    name: str
    duration: list[int]
