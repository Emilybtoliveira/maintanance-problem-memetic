import json
import os
import sys
import time
import concurrent.futures
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from datamodels.problem import Problem
from datamodels.input_problem_loader import InputProblemLoader
from utils.log import log
from solvers.memetic import MemeticAlgorithm
from solvers.gurobi import Gurobi


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

    log(instance, f"Starting at {current_time}", first_call=True)

    problem_loader = InputProblemLoader(input_path)

    problem = problem_loader()

    return problem


def make_optimization(
    instance,
    problem,
    pop_size,
    crossover_rate,
    mutation_rate,
    target=None,
    time_limit=TIME_LIMIT,
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

    gb = Gurobi(problem=problem, time_limit=300, threads=1)
    gb_pool_solutions = gb.optimize()
    log(f"{instance}", f"Gurobi solution: {gb.get_objective_value()}")

    elapsed = time.time() - start_time_execution
    remaining_time = time_limit - elapsed

    if remaining_time <= 10:  # less than 10s left
        remaining_time = 60
        log(f"{instance}", "Warning: remaining_time adjusted to 60 seconds")

    log(f"{instance}", f"Running Memetic Algorithm for {remaining_time:.0f} seconds...")

    solution, objective_value, solution_formated = MemeticAlgorithm(
        file_name=instance,
        problem=problem,
        gb_pool_solutions=gb_pool_solutions,
        pop_size=pop_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        remaining_time=remaining_time,
        target=target,
    ).optimize()

    log(f"{instance}", "\nOptimization completed.")
    log(f"{instance}", "Saving the solution to the output file...")

    total_time_execution = time.time() - start_time_execution

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{instance}.txt"
    with open(output_file, "w") as f:
        f.write(solution_formated)

    log(f"{instance}", "Done!\n")

    return solution, objective_value, total_time_execution


def run_vasquez_solver(instance, time_limit, target=0):
    instance_full_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/../input/" + instance + ".json"
    )
    print(instance_full_path)
    cmd = [
        os.path.dirname(os.path.abspath(__file__)) + "/../rc/challengeRTE",
        "-p",
        instance_full_path,
        "-t",
        str(time_limit),
        "-target",
        str(target),
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()

    print(result)
    elapsed = end - start
    return elapsed, result.stdout


def generate_perfomance_profile(pop_size, crossover_rate, mutation_rate):

    print("Generating performance profile...")
    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    times = {"Vasquez": [], "Memetic": []}

    N_RUNS = 1
    INSTANCES = {  # instance : target
        "A_07": 2273,
        "A_08": 745,
        "A_09": 1508,
        "B_07": 7889,
        "B_08": 7531,
        "B_14": 12213,
        "C_02": 3646,
        "C_06": 8688,
        "C_12": 13180,
    }
    time_limit = 900

    for inst, target in INSTANCES.items():
        print(f"==> Rodando instância {inst}")
        vasquez_tt = np.mean(
            [run_vasquez_solver(inst, time_limit, target)[0] for _ in range(N_RUNS)]
        )
        times["Vasquez"].append(vasquez_tt)

        problem = load_problem(os.path.dirname(os.path.abspath(__file__)), inst)
        mem_t = np.mean(
            [
                make_optimization(
                    inst,
                    problem,
                    pop_size,
                    crossover_rate,
                    mutation_rate,
                    target,
                    time_limit,
                )[2]
                for _ in range(N_RUNS)
            ]
        )
        times["Memetic"].append(mem_t)

    T = np.vstack([times["Vasquez"], times["Memetic"]]).T

    min_times = np.min(T, axis=1)
    ratios = T / min_times[:, None]

    taus = np.linspace(1, np.max(ratios) + 0.5, 100)
    plt.figure(figsize=(6, 4))

    for j, algo in enumerate(times.keys()):
        rho = [np.mean(ratios[:, j] <= tau) for tau in taus]
        plt.plot(taus, rho, label=algo)

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\rho_s(\tau)$")
    plt.title("Performance Profile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/performance_profile.png", dpi=300)
    plt.show()


def generate_ttt_plot(pop_size, crossover_rate, mutation_rate):

    print("Generating TTT-plots...")
    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    N_RUNS = 30
    INSTANCES = {  # instance : target
        "A_07": 2281,
        "A_08": 746,
        "A_09": 1509,
    }
    time_limit = 600

    all_vasquez_times = {}
    all_memetic_times = {}

    for inst, target in INSTANCES.items():
        vasquez_times = []
        memetic_times = []

        print(f"==> Rodando instância {inst}")

        for i in range(N_RUNS):
            t_vasquez, _ = run_vasquez_solver(inst, time_limit, target)
            vasquez_times.append(t_vasquez)

            problem = load_problem(os.path.dirname(os.path.abspath(__file__)), inst)
            t_mem = make_optimization(
                inst,
                problem,
                pop_size,
                crossover_rate,
                mutation_rate,
                target,
                time_limit,
            )[2]
            memetic_times.append(t_mem)

        all_vasquez_times[inst] = vasquez_times
        all_memetic_times[inst] = memetic_times

        cpp_sorted = np.sort(vasquez_times)
        mem_sorted = np.sort(memetic_times)
        p = np.linspace(0, 1, len(cpp_sorted))

        plt.figure(figsize=(7, 5))
        plt.plot(cpp_sorted, p, label="Vasquez (C++)", linestyle="--")
        plt.plot(mem_sorted, p, label="Memetic (Python)", linestyle="-")

        plt.xlabel("Tempo (s)")
        plt.ylabel("Proporção de execuções")
        plt.title(f"TTT-Plot — Instância {inst} (target {target})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("output", exist_ok=True)
        plt.savefig(f"output/ttt_plot_{inst}.png", dpi=300)
        plt.close()

    print("TTT-plots gerados para todas as instâncias!")


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
        _, fitness, _ = make_optimization(
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

    # print("Instances to be run:", instances)

    # breakpoint()

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
            instance = "A_02"  # The default instance because it is the smallest and runs faster
            pop_size = algorithm_parameters["pop_size"]
            crossover_rate = algorithm_parameters["crossover_rate"]
            mutation_rate = algorithm_parameters["mutation_rate"]

        problem = load_problem(current_dir, instance)

        # ------------- Make the Optimization ----------------
        # start_time_execution = time.time()

        # gb = Gurobi(problem=problem, time_limit=300, threads=1)

        # gb_pool_solutions = gb.optimize()

        # remaining_time = TIME_LIMIT - (time.time() - start_time_execution)

        # print(f"Running Memetic Algorithm for {remaining_time:.0f} seconds...")

        # _, objective_value, solution_formated = MemeticAlgorithm(
        #     file_name=instance,
        #     problem=problem,
        #     gb_pool_solutions=gb_pool_solutions,
        #     pop_size=pop_size,
        #     crossover_rate=crossover_rate,
        #     mutation_rate=mutation_rate,
        #     remaining_time=remaining_time,
        # ).optimize()

        # print(solution_formated)
        # print(f"{instance}: {objective_value}")

        _, objective_value, total_time_execution = make_optimization(
            instance=instance,
            problem=problem,
            pop_size=pop_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            time_limit=TIME_LIMIT,
        )

        print(f"{instance}: {objective_value} in {total_time_execution:.2f} seconds")

    # breakpoint()

    # # save the logs folder in a backup folder with timestamp
    # timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # backup_dir = Path(current_dir).parent / "logs_backup" / timestamp
    # backup_dir.mkdir(parents=True, exist_ok=True)
    # logs_dir = Path(current_dir).parent / "logs"
    # if logs_dir.exists():
    #     for log_file in logs_dir.glob("*.txt"):
    #         log_file.rename(backup_dir / log_file.name)

    # # save the output folder in a backup folder with timestamp
    # output_dir = Path(current_dir).parent / "output"
    # backup_output_dir = Path(current_dir).parent / "output_backup" / timestamp
    # backup_output_dir.mkdir(parents=True, exist_ok=True)
    # if output_dir.exists():
    #     for output_file in output_dir.glob("*.txt"):
    #         output_file.rename(backup_output_dir / output_file.name)

    # Generate performance profile and TTT plots

    # breakpoint()

    # generate_perfomance_profile(
    #     algorithm_parameters["pop_size"],
    #     algorithm_parameters["crossover_rate"],
    #     algorithm_parameters["mutation_rate"],
    # )

    # generate_ttt_plot(
    #     algorithm_parameters["pop_size"],
    #     algorithm_parameters["crossover_rate"],
    #     algorithm_parameters["mutation_rate"],
    # )


if __name__ == "__main__":
    main()
