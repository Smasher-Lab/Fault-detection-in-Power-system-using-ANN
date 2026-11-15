class CSP:
    def __init__(self, variables, domains, neighbors):
        self.variables = variables          # List of variables
        self.domains = domains              # Dict mapping variable -> list of legal values
        self.neighbors = neighbors          # Dict mapping variable -> list of adjacent variables
        self.constraints = {}

    def is_consistent(self, var, value, assignment):
        """Checks if assigning `value` to `var` violates any constraints with already assigned neighbors."""
        for neighbor in self.neighbors[var]:
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True

    def backtrack(self, assignment={}):
        """Core Backtracking Search Algorithm."""
        # Base Case: All variables assigned
        if len(assignment) == len(self.variables):
            return assignment

        # 1. Select an unassigned variable (Minimum Remaining Values / MRV heuristic)
        unassigned = [v for v in self.variables if v not in assignment]
        var = min(unassigned, key=lambda v: len(self.domains[v]))

        # 2. Try values in the variable's domain
        for value in self.domains[var]:
            if self.is_consistent(var, value, assignment):
                assignment[var] = value  # Assign value

                # Recursive Step
                result = self.backtrack(assignment)
                if result is not None:
                    return result

                # Backtrack: Remove assignment if it led to a failure
                del assignment[var]

        return None  # Return None if no valid assignment is possible


# ==========================================
# Example Usage: Map Coloring Problem
# ==========================================
if __name__ == "__main__":
    # 1. Define Variables (Australian States)
    variables = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]

    # 2. Define Domains (Available Colors for each variable)
    domains = {var: ["Red", "Green", "Blue"] for var in variables}

    # 3. Define Adjacency (Neighbors/Constraints)
    neighbors = {
        "WA": ["NT", "SA"],
        "NT": ["WA", "SA", "Q"],
        "SA": ["WA", "NT", "Q", "NSW", "V"],
        "Q":  ["NT", "SA", "NSW"],
        "NSW": ["Q", "SA", "V"],
        "V":  ["SA", "NSW"],
        "T":  []  # Tasmania is an island (no neighbors)
    }

    # Initialize and Solve
    csp_solver = CSP(variables, domains, neighbors)
    solution = csp_solver.backtrack()

    # Print Results
    print("--- CSP Solution ---")
    if solution:
        for var, color in solution.items():
            print(f"{var}: {color}")
    else:
        print("No solution found.")