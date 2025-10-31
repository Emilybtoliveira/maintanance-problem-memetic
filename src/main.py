import json
import os
import sys
from pathlib import Path
import time
import concurrent.futures

sys.path.append(str(Path(__file__).resolve().parent.parent))

from datamodels.problem import Problem

from datamodels.input_problem_loader import InputProblemLoader

from utils.log import log

TIME_LIMIT = 60 * 15  # 15 minutes


def load_problem(current_dir, instance) -> Problem:
    """
    Load the problem from the input file

    Args:
        current_dir (str): The current directory.
        instance (str): The instance name.
    Returns:
        dict: The problem object.
    """

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    input_path = os.path.join(current_dir, f"../input/{instance}.json")

    log(instance, f"Starting at {current_time}")

    problem_loader = InputProblemLoader(input_path)

    problem = problem_loader()

    return problem


def make_optimization(
    instance,
    problem,
    pop_size,
    crossover_rate,
    mutation_rate,
) -> tuple:
    """
    Perform optimization on the given problem instance using specified parameters.

    Args:
        instance (str): The name of the problem instance.
        problem (Problem): The problem to be optimized.
        pop_size (int): The population size.
        crossover_rate (float): The crossover rate.
        mutation_rate (float): The mutation rate.
    Returns:
        A tuple containing the solution and its fitness value.
    """
    start_time_execution = time.time()
    log(f"{instance}", "Optimizing the problem...")

    optimization_step = OptimizationStep(
        start_time_execution=start_time_execution,
        time_limit=TIME_LIMIT,
        problem=problem,
        file_name=instance,
        pop_size=pop_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
    )

    optimization_info = optimization_step()

    log(f"{instance}", "\nOptimization completed.")
    log(f"{instance}", "Saving the solution to the output file...")

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{instance}.txt"
    with open(output_file, "w") as f:
        f.write(f"{optimization_info['solution']}")

    log(f"{instance}", "Done!\n")

    return optimization_info["solution"], optimization_info["objective_value"]


def run_all_instances(instances, algorithm_parameters) -> None:
    """
    Run all instances

    Args:
        instances (list): The list of instances
        algorithm_parameters (dict): The parameters of the instances
    Returns:
        None
    """
    for instance in instances:
        problem = load_problem(os.path.dirname(os.path.abspath(__file__)), instance)

        _, fitness = make_optimization(
            instance=instance,
            problem=problem,
            pop_size=algorithm_parameters["pop_size"],
            crossover_rate=algorithm_parameters["crossover_rate"],
            mutation_rate=algorithm_parameters["mutation_rate"],
        )

        print(f"{instance}: {fitness}")


def run_all_instances_parallel(instances, algorithm_parameters) -> None:
    """
    Run instances in parallel with concurrent.futures

    Args:
        instances (list): The list of instances
        algorithm_parameters (dict): The parameters of the instances
    Returns:
        None
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = {
            executor.submit(
                make_optimization,
                instance_name,
                load_problem(os.path.dirname(os.path.abspath(__file__)), instance_name),
                algorithm_parameters["pop_size"],
                algorithm_parameters["crossover_rate"],
                algorithm_parameters["mutation_rate"],
            ): instance_name
            for instance_name in instances
        }

        for future in concurrent.futures.as_completed(results):
            instance = results[future]
            try:
                _, fitness = future.result()
                print(f"{instance}: {fitness}")
            except Exception as e:
                print(f"{instance} generated an exception: {e}")


def main() -> None:
    # ------------- Load the Problem ----------------

    current_dir = os.path.dirname(os.path.abspath(__file__))

    parameters_path = os.path.join(current_dir, f"../input/parameters.json")

    try:
        with open(parameters_path, "r") as file:
            parameters = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {parameters_path} not found")

    irace = parameters["irace"]  # If True, the parameters will be passed by irace
    run_all = parameters["run_all"]
    parallel = parameters["parallel"]
    algorithm_parameters = parameters["algorithm_parameters"]

    # List all instances in the input folder except the parameters file
    input_dir = Path(current_dir).parent / "input"
    instances = [
        p.stem
        for p in input_dir.glob("*.json")
        if p.stem != "parameters" and p.stem != "E_01" and p.stem != "E_02"
    ]
    instances = sorted(instances)

    if run_all:
        if parallel:
            run_all_instances_parallel(instances, algorithm_parameters)
        else:
            run_all_instances(instances, algorithm_parameters)
    else:
        if irace:
            instance = sys.argv[1]
            pop_size = int(sys.argv[2])
            crossover_rate = float(sys.argv[3])
            mutation_rate = float(sys.argv[4])
        else:
            instance = "A_09"  # The default instance because it is the smallest and runs faster
            pop_size = algorithm_parameters["pop_size"]
            crossover_rate = algorithm_parameters["crossover_rate"]
            mutation_rate = algorithm_parameters["mutation_rate"]

        problem = load_problem(current_dir, instance)

        # ------------- Make the Optimization ----------------

        # _, objective_value = make_optimization(
        #     instance=instance,
        #     problem=problem,
        #     pop_size=pop_size,
        #     crossover_rate=crossover_rate,
        #     mutation_rate=mutation_rate,
        # )

        # print(f"{instance}: {objective_value}")


if __name__ == "__main__":
    main()
