def format_solution(interventions, best):
    solution = []
    for i, start_time in enumerate(best):
        intervention_name = interventions[i].name
        solution.append(f"{intervention_name} {int(start_time)}")
    return "\n".join(solution)
