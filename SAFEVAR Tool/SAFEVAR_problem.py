from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class CarlaProblem(FloatProblem):
    def __init__(self, runner):
        super().__init__()
        self.number_of_variables = 12
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.lower_bound = [4200, 0.1, 1.0, 0.2, 0.3, 8.0, 1940, 0.2, 1.0, 0.2, 31.7, 1200]
        self.upper_bound = [5900, 0.2, 3.0, 0.4, 0.6, 12.0, 2700, 0.5, 3.0, 0.3, 36.0, 1600]

        self.runner = runner

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        variables = solution.variables

        # Map variables to physics parameters
        physics = [round(v, 3) for v in variables]

        # Run simulation and calculate objectives
        results = self.runner.run(physics)
        max_change_ratio = self._calculate_max_change_ratio(physics)
        modified_params = self._count_modified_parameters(physics)

        # Assign objectives
        solution.objectives[0] = max_change_ratio
        solution.objectives[1] = results
        solution.objectives[2] = modified_params

        return solution

    def _calculate_max_change_ratio(self, physics):
        # Placeholder for change ratio logic
        return max(abs((v - target) / (ub - lb))
                   for v, target, lb, ub in zip(physics, [5800, 0.15, 2, 0.35, 0.5, 10, 2404, 0.3, 2, 0.25, 31.7, 1500], 
                                                self.lower_bound, self.upper_bound))

    def _count_modified_parameters(self, physics):
        # Placeholder for counting significant changes
        return sum(1 for v, target in zip(physics, [5800, 0.15, 2, 0.35, 0.5, 10, 2404, 0.3, 2, 0.25, 31.7, 1500])
                   if abs(v - target) > 0.01)

    def get_name(self):
        return 'CarlaProblem'