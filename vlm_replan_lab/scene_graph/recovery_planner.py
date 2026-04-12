class RecoveryPlanner:

    def generate(self,graph):

        plan = []

        for name,data in graph.items():

            if data["upright"] == False:

                plan.append(("recover",name))

                plan.append(("place","basket"))

        return plan