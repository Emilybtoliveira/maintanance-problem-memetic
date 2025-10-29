import json
import time
import numpy as np

from preprocessing.model.resource import Resource
from preprocessing.model.exclusion import Exclusion
from preprocessing.model.intervention import Intervention
from preprocessing.model.problem import Problem
from preprocessing.model.season import Season
from preprocessing.model.time_horizon import TimeHorizon
from utils.log import log


class InputProblemLoader:
    """
    A class responsible for loading an input problem from a file.

    Args:
        path (str): The path to the input problem file.

    Raises:
        FileNotFoundError: If the specified file is not found.

    """

    def __init__(self, path: str):
        self.path = path

    def __call__(self) -> dict:
        """
        Loads the input problem from the specified file.

        Returns:
            dict: The loaded problem as a dictionary.

        Raises:
            FileNotFoundError: If the specified file is not found.

        """
        return self._execute()

    def _parse(self) -> dict:
        try:
            with open(self.path, "r") as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.path} not found")

    def _get_resources(self, input_data: dict) -> np.ndarray[Resource]:
        """
        Get a list of Resource objects from the given input data.

        Args:
            data (dict): The input data containing resource information.

        Returns:
            list[Resource]: A list of Resource objects.

        """
        resources_data = input_data["Resources"]
        resource_names = list(resources_data.keys())
        resources = np.array([])

        for index, resource in enumerate(resources_data.values()):
            resources = np.append(
                resources,
                Resource(
                    name=resource_names[index],
                    max=resource["max"],
                    min=resource["min"],
                ),
            )

        return resources

    def _get_interventions(
        self, input_data: dict, resources: list[Resource]
    ) -> np.ndarray[Intervention]:
        """
        Get a list of Intervention objects based on the input data and resources.

        Args:
            input_data (dict): The input data containing information about interventions.
            resources (list[Resource]): The list of available resources.

        Returns:
            list[Intervention]: A list of Intervention objects.

        """
        interventions_data = input_data["Interventions"]
        intervention_names = list(interventions_data.keys())
        interventions = np.array([])

        for index, intervention in enumerate(interventions_data.values()):
            interventions = np.append(
                interventions,
                Intervention(
                    name=intervention_names[index],
                    tmax=intervention["tmax"],
                    delta=intervention["Delta"],
                    resource_workload=intervention["workload"],
                    risk=intervention["risk"],
                ),
            )

        return interventions

    def _get_season_duration(self, season: dict, seasons: str) -> int:
        if season in seasons:
            return np.array([int(s) for s in seasons[season]])
        else:
            raise ValueError(f"Season {season} not found in the input data")

    def _get_season(self, input_data: dict, seasons: dict) -> Season:
        season_name = input_data[-1]
        season = Season(
            name=season_name,
            duration=self._get_season_duration(season_name, seasons),
        )
        return season

    def _get_exclusions(
        self, input_data: dict, interventions: list[Intervention]
    ) -> np.ndarray[Exclusion]:
        """
        Get the list of exclusions based on the input data and interventions.

        Args:
            input_data (dict): The input data containing exclusions information.
            interventions (list[Intervention]): The list of interventions.

        Returns:
            list[Exclusion]: The list of exclusions.

        """
        exclusions_data = input_data["Exclusions"]
        seasons = input_data["Seasons"]
        exclusions_names = list(exclusions_data.keys())
        exclusions = np.array([])

        for index, exclusion in enumerate(exclusions_data.values()):

            exclusions = np.append(
                exclusions,
                Exclusion(
                    name=exclusions_names[index],
                    interventions=[
                        intervention.name
                        for intervention in interventions
                        if intervention.name in exclusion
                    ],
                    season=self._get_season(exclusion, seasons),
                ),
            )

        return exclusions

    def _get_time_horizon(self, input_data: dict) -> TimeHorizon:
        """
        Extracts the time horizon from the input data.

        Args:
            input_data (dict): The input data containing the time horizon information.

        Returns:
            TimeHorizon: An instance of the TimeHorizon class representing the extracted time horizon.
        """
        time_horizon_data = input_data["T"]
        return TimeHorizon(
            time_steps=time_horizon_data,
        )

    def _get_scenarios(self, input_data: dict) -> list[int]:
        """
        Get the list of scenarios from the input data.

        Parameters:
            input_data (dict): The input data containing the scenarios.

        Returns:
            list[int]: The list of scenarios as integers.
        """
        scenarios_data = input_data["Scenarios_number"]
        return np.array([int(scenario) for scenario in scenarios_data])

    def _get_quantile(self, input_data: dict) -> float:
        """
        Get the quantile value from the input data.

        Parameters:
            input_data (dict): The input data containing the quantile value.

        Returns:
            float: The quantile value as a float.
        """
        return float(input_data["Quantile"])

    def _get_alpha(self, input_data: dict) -> float:
        """
        Get the alpha value from the input data.

        Parameters:
            input_data (dict): The input data containing the alpha value.

        Returns:
            float: The alpha value as a float.
        """
        return float(input_data["Alpha"])

    def _get_computation_time(self, input_data: dict) -> float:
        """
        Get the computation time from the input data.

        Parameters:
            input_data (dict): The input data containing the computation time.

        Returns:
            float: The computation time as a float.
        """
        try:
            return float(input_data["ComputationTime"])
        except KeyError:
            return None

    def _execute(self) -> Problem:
        """
        Execute the input problem loading process and return the loaded problem.

        Args:
            None

        Returns:
            Problem: The loaded problem.
        """
        start_time = time.time()

        file_name = self.path.split("/")[-1].split(".")[0]

        log(f"{file_name}", "Loading the problem...")

        data = self._parse()

        resources = self._get_resources(input_data=data)
        interventions = self._get_interventions(input_data=data, resources=resources)
        exclusions = self._get_exclusions(input_data=data, interventions=interventions)
        time_horizon = self._get_time_horizon(input_data=data)
        scenarios = self._get_scenarios(input_data=data)
        quantile = self._get_quantile(input_data=data)
        alpha = self._get_alpha(input_data=data)
        computation_time = self._get_computation_time(input_data=data)

        problem = Problem(
            resources=resources,
            interventions=interventions,
            exclusions=exclusions,
            time_horizon=time_horizon,
            scenarios=scenarios,
            quantile=quantile,
            alpha=alpha,
            computation_time=computation_time,
        )

        total_time = time.time() - start_time

        log(f"{file_name}", "Problem loaded successfully!")
        log(f"{file_name}", f"Elapsed time: {total_time:.2f} seconds")

        return problem
