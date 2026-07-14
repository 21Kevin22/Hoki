class SymbolicPlanner:

    def generate_initial_plan(self):

        plan = []

        for i in range(5):

            plan.append(("pick",f"mug_{i}"))

            plan.append(("place","basket"))

        return plan