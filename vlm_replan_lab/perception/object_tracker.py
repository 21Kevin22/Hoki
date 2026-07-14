from omni.isaac.core.utils.stage import get_current_stage


class ObjectTracker:

    def track(self):

        stage = get_current_stage()

        objects = []

        for prim in stage.Traverse():

            name = prim.GetName()

            if "mug" in name:

                pose = prim.GetAttribute("xformOp:translate").Get()

                rot = prim.GetAttribute("xformOp:orient").Get()

                objects.append(
                    {
                        "name": name,
                        "position": pose,
                        "orientation": rot
                    }
                )

        return objects