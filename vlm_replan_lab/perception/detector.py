import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom


class Detector:
    def _get_pose(self, path):
        prim = get_prim_at_path(path)
        if prim is None:
            return None, None

        xform = UsdGeom.Xformable(prim)
        mat = xform.ComputeLocalToWorldTransform(0)

        pos = mat.ExtractTranslation()
        quat = mat.ExtractRotationQuat()

        return np.array([pos[0], pos[1], pos[2]], dtype=float), quat

    def detect(self, mug_names, basket_name):
        objects = []

        for mug in mug_names:
            pos, quat = self._get_pose(f"/World/{mug}")
            if pos is None:
                continue

            objects.append({
                "name": mug,
                "position": pos,
                "orientation": quat,
                "type": "mug",
            })

        basket_pos, basket_quat = self._get_pose("/World/basket/base")
        if basket_pos is not None:
            objects.append({
                "name": basket_name,
                "position": basket_pos,
                "orientation": basket_quat,
                "type": "basket",
            })

        return objects