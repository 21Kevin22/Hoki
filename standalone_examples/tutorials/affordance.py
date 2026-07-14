import os
import numpy as np
from isaacsim import SimulationApp
config = {"headless": True, "renderer": "RayTracedLighting", "width": 1024, "height": 1024}
simulation_app = SimulationApp(config)

from pxr import Usd, UsdGeom, Gf
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import get_current_stage

enable_extension("omni.replicator.core")

output_dir = os.path.join(os.getcwd(), "beer_mug_ring_handle_mesh")
os.makedirs(output_dir, exist_ok=True)

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

def create_torus_mesh(stage, path, major_r=0.07, minor_r=0.012, seg_major=48, seg_minor=16):
    """PXRL mesh でトーラスを生成し、ステージに配置する。"""
    mesh = UsdGeom.Mesh.Define(stage, path)

    verts = []
    normals = []
    for i in range(seg_major):
        theta = 2.0 * np.pi * i / seg_major
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        center = np.array([0.0, major_r * cos_t, major_r * sin_t])
        for j in range(seg_minor):
            phi = 2.0 * np.pi * j / seg_minor
            cos_p, sin_p = np.cos(phi), np.sin(phi)
            normal = np.array([sin_p, cos_p * cos_t, cos_p * sin_t])
            point = center + minor_r * normal
            verts.append(Gf.Vec3f(*point))
            normals.append(Gf.Vec3f(*normal))

    faces = []
    counts = []
    for i in range(seg_major):
        i_next = (i + 1) % seg_major
        for j in range(seg_minor):
            j_next = (j + 1) % seg_minor
            v0 = i * seg_minor + j
            v1 = i_next * seg_minor + j
            v2 = i_next * seg_minor + j_next
            v3 = i * seg_minor + j_next
            # quad -> two triangles
            faces.extend([v0, v1, v2, v0, v2, v3])
            counts.extend([3, 3])

    mesh.CreatePointsAttr(verts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation("vertex")
    return mesh

def create_beer_mug():
    stage = get_current_stage()
    mug_root = "/World/BeerMug"
    UsdGeom.Xform.Define(stage, mug_root)

    # 本体
    body_path = f"{mug_root}/Body"
    body = UsdGeom.Cylinder.Define(stage, body_path)
    body.CreateHeightAttr(0.24)
    body.CreateRadiusAttr(0.042)
    body.CreateAxisAttr("Z")
    body_prim = XFormPrim(body_path)
    body_prim.set_world_poses(positions=np.array([[0.0, 0.0, 0.12]]))
    body_prim.set_local_scales(np.array([[0.92, 0.92, 1.0]]))
    rep.modify.semantics(semantics=[("class", "grasp-body")], input_prims=[body_path])

    # 厚底
    base_path = f"{mug_root}/Base"
    base = UsdGeom.Cylinder.Define(stage, base_path)
    base.CreateHeightAttr(0.02)
    base.CreateRadiusAttr(0.05)
    base.CreateAxisAttr("Z")
    base_prim = XFormPrim(base_path)
    base_prim.set_world_poses(positions=np.array([[0.0, 0.0, 0.01]]))
    rep.modify.semantics(semantics=[("class", "grasp-body")], input_prims=[base_path])

    # 取っ手（メッシュのトーラス）
    handle_path = f"{mug_root}/Handle"
    create_torus_mesh(stage, handle_path, major_r=0.04, minor_r=0.004, seg_major=48, seg_minor=16)
    handle_prim = XFormPrim(handle_path)
    handle_prim.set_world_poses(
        positions=np.array([[0.07, 0.0, 0.12]]),           # ボディ中心高さに合わせ側方へ
        orientations=np.array([[0.7071, 0, 0, 0.7071]])    # Y軸90°回転で外向き
    )
    rep.modify.semantics(semantics=[("class", "grasp-handle")], input_prims=[handle_path])

    # 液体
    liquid_path = f"{mug_root}/Liquid"
    liquid = UsdGeom.Cylinder.Define(stage, liquid_path)
    liquid.CreateHeightAttr(0.18)
    liquid.CreateRadiusAttr(0.039)
    liquid.CreateAxisAttr("Z")
    liquid_prim = XFormPrim(liquid_path)
    liquid_prim.set_world_poses(positions=np.array([[0.0, 0.0, 0.10]]))
    rep.modify.semantics(semantics=[("class", "contain-liquid")], input_prims=[liquid_path])

create_beer_mug()

set_camera_view(eye=[0.35, 0.35, 0.35], target=[0.0, 0.0, 0.12])

render_product = rep.create.render_product("/OmniverseKit_Persp", (1024, 1024))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=output_dir,
    rgb=True,
    semantic_segmentation=True,
    colorize_semantic_segmentation=True
)
writer.attach([render_product])

my_world.reset()
for i in range(50):
    my_world.step(render=True)
    if i % 10 == 0:
        simulation_app.update()

for i in range(20):
    my_world.step(render=True)
    rep.orchestrator.step()

print(f"=== 完了: {output_dir} ===")
simulation_app.close()
