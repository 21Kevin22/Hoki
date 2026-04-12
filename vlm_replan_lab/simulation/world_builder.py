import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import XFormPrim


class WorldBuilder:
    def __init__(self, world):
        self.world = world
        self.stage = get_current_stage()
        self.mug_names = [f"mug{i}" for i in range(5)]
        self.basket_name = "basket"

    def build(self):
        self._build_lighting()
        self._build_table()
        self._build_basket()
        self._build_mugs()

    def _apply_color(self, prim, color):
        mat_path = prim.GetPath().pathString + "_Mat"
        material = UsdShade.Material.Define(self.stage, mat_path)
        shader = UsdShade.Shader.Define(self.stage, mat_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(*color))
        shader.CreateInput(
            "roughness", Sdf.ValueTypeNames.Float
        ).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    def _build_lighting(self):
        dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1200.0)

    def _build_table(self):
        path = "/World/table"
        cube = UsdGeom.Cube.Define(self.stage, path)
        XFormPrim(path).set_world_poses(
            positions=np.array([[0.45, 0.0, 0.45]])
        )
        XFormPrim(path).set_local_scales(
            np.array([[0.55, 0.35, 0.05]])
        )
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        self._apply_color(cube.GetPrim(), (0.45, 0.28, 0.18))

    def _build_basket(self):
        base_path = "/World/basket/base"
        base = UsdGeom.Cube.Define(self.stage, base_path)
        XFormPrim(base_path).set_world_poses(
            positions=np.array([[0.72, 0.0, 0.53]])
        )
        XFormPrim(base_path).set_local_scales(
            np.array([[0.12, 0.12, 0.015]])
        )
        UsdPhysics.CollisionAPI.Apply(base.GetPrim())
        self._apply_color(base.GetPrim(), (0.3, 0.5, 0.2))

        walls = [
            ("/World/basket/front", [0.72, -0.12, 0.59], [0.12, 0.01, 0.06]),
            ("/World/basket/back",  [0.72,  0.12, 0.59], [0.12, 0.01, 0.06]),
            ("/World/basket/left",  [0.60,  0.00, 0.59], [0.01, 0.12, 0.06]),
            ("/World/basket/right", [0.84,  0.00, 0.59], [0.01, 0.12, 0.06]),
        ]

        for path, pos, scale in walls:
            cube = UsdGeom.Cube.Define(self.stage, path)
            XFormPrim(path).set_world_poses(
                positions=np.array([pos])
            )
            XFormPrim(path).set_local_scales(
                np.array([scale])
            )
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            self._apply_color(cube.GetPrim(), (0.3, 0.5, 0.2))

    def _build_mugs(self):
        colors = [
            (0.9, 0.2, 0.2),
            (0.2, 0.9, 0.2),
            (0.2, 0.2, 0.9),
            (0.9, 0.9, 0.2),
            (0.2, 0.9, 0.9),
        ]

        positions = [
            [0.20,  0.20, 0.56],
            [0.30,  0.10, 0.56],
            [0.40,  0.00, 0.56],
            [0.30, -0.10, 0.56],
            [0.20, -0.20, 0.56],
        ]

        for i, (pos, color) in enumerate(zip(positions, colors)):
            body_path = f"/World/mug{i}"
            cyl = UsdGeom.Cylinder.Define(self.stage, body_path)
            cyl.CreateRadiusAttr(0.035)
            cyl.CreateHeightAttr(0.08)
            cyl.CreateAxisAttr("Z")

            XFormPrim(body_path).set_world_poses(
                positions=np.array([pos])
            )

            UsdPhysics.RigidBodyAPI.Apply(cyl.GetPrim())
            UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
            UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateMassAttr(0.20)

            self._apply_color(cyl.GetPrim(), color)