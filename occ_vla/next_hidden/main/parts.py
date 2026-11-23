"""Ground-truth container parts per task (PREREG_main.md §3 roles, §4 L1). Used by the collision metric for every
condition, and by the GT-role filters (C4, C8). Part labels come from MuJoCo geometry only (no per-task hand edits
beyond naming which container / drawer the task uses):
  handle : drawer handle boxes (StowSafe drawer_parts) / microwave door capsules
  panel  : drawer front box(es) / microwave door box
  floor  : horizontal plates (thin axis vertical) inside the task container (drawer bottom, microwave floor,
           caddy and basket bottoms) -- where a held object may be put down
  wall   : every other collision geom of the task container
Other containers' geoms (other drawers, the cabinet carcass) are 'wall' too: nothing may touch them."""
import numpy as np
import mujoco

ROBOT = ("robot0", "gripper0", "mount0")
HAND = ("gripper0",)                       # palm + fingers (+ eef) bodies
TASK_CONTAINER = {"T1": ("microwave_1", None), "T2": ("white_cabinet_1", "bottom"), "T3": ("wooden_cabinet_1", "top"),
                  "T4": ("wooden_cabinet_1", "middle"), "T5": ("desk_caddy_1", None),
                  "T6": ("wooden_cabinet_1", "top"), "T7": ("basket_1", None), "T8": ("basket_1", None),
                  "T9": ("flat_stove_1", None), "T10": ("plate_1", None), "T11": ("plate_1", None),
                  "T12": ("basket_1", None), "T13": ("flat_stove_1", None)}
TARGETS = {"T1": ["white_yellow_mug_1"], "T2": ["akita_black_bowl_1"], "T3": ["akita_black_bowl_1"], "T4": [],
           "T5": ["black_book_1"], "T6": ["akita_black_bowl_1"], "T7": ["alphabet_soup_1", "tomato_sauce_1"],
           "T8": ["cream_cheese_1", "butter_1"], "T9": ["moka_pot_1"], "T10": ["porcelain_mug_1", "white_yellow_mug_1"],
           "T11": ["porcelain_mug_1", "chocolate_pudding_1"], "T12": ["alphabet_soup_1", "cream_cheese_1"],
           "T13": ["moka_pot_1", "moka_pot_2"]}
DEST_EXTRA = {"T6": ["plate_1"], "T10": ["plate_2"]}
# destination objects a held object may touch (besides container floors)
PART_ALL = {"T9": "panel", "T13": "panel", "T10": "floor", "T11": "floor"}   # PREREG 2026-11-23 (2): stove / plates


def _collides(m, g):
    return bool(m.geom_contype[g] or m.geom_conaffinity[g])


def _thin_axis_vertical(m, d, g):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
        return False
    s = m.geom_size[g]; k = int(np.argmin(s))
    if s[k] > 0.25 * np.sort(s)[1]:
        return False
    return abs(d.geom_xmat[g].reshape(3, 3)[2, k]) > 0.9


class TaskParts:
    def __init__(self, task, model, data):
        from stowsafe.roles import drawer_parts  # noqa: F401  (part split of drawers; reused logic below)
        m, d = model, data
        self.task = task
        root, level = TASK_CONTAINER[task]
        bodies = {b for b in range(m.nbody) if m.body(m.body_rootid[b]).name.startswith(root)
                  or m.body(b).name.startswith(root)}
        self.part = {}                                  # geom id -> handle|panel|floor|wall
        task_body = None
        if level is not None:
            task_body = m.body(f"{root}_cabinet_{level}").id
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b not in bodies or not _collides(m, g):
                continue
            name = m.body(b).name
            if "microwave" in root:
                if name.endswith("microdoorroot"):
                    self.part[g] = "handle" if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else "panel"
                else:
                    self.part[g] = "wall"
            elif level is not None:
                self.part[g] = "wall"
            else:                                       # caddy, basket: plates are floors
                self.part[g] = "floor" if _thin_axis_vertical(m, d, g) else "wall"
        if task in PART_ALL:                            # stove: knob + burner all "panel"; plates: all "floor"
            self.part = {g: PART_ALL[task] for g in self.part}
        if "microwave" in root:                         # microwave floor = the lowest horizontal plate of the body
            plates = [g for g, p in self.part.items() if p == "wall" and _thin_axis_vertical(m, d, g)]
            if plates:
                self.part[min(plates, key=lambda g: d.geom_xpos[g][2])] = "floor"
        if task_body is not None:                       # the task's drawer: handle / front / interior (floor + walls)
            self._label_drawer(m, d, task_body)
        self.targets = {m.body(n + "_main").id if n + "_main" in [m.body(i).name for i in range(m.nbody)]
                        else m.body(n).id for n in TARGETS[task]}
        self.target_roots = set(TARGETS[task])
        self.dest_roots = set(DEST_EXTRA.get(task, []))
        self.container_root = root
        self.joint = None if level is None and "microwave" not in root else (
            m.joint(f"{root}_{level}_level").id if level else m.joint(f"{root}_microjoint").id)

    def _label_drawer(self, m, d, b):
        """Same rule as StowSafe roles.drawer_parts (pull direction from the slide joint): handle = small boxes beyond
        the front, front = the big box(es) at the front; the rest is interior -> floor (horizontal plates) / wall."""
        j = next(j for j in range(m.njnt) if m.jnt_bodyid[j] == b)
        lo, hi = m.jnt_range[j]
        pull = m.jnt_axis[j] * (np.sign(lo) if abs(lo) > abs(hi) else np.sign(hi))
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and _collides(m, g)
                 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        along = {g: float(np.asarray(m.geom_pos[g]) @ pull) for g in geoms}
        vol = {g: float(np.prod(m.geom_size[g])) for g in geoms}
        big = [g for g in geoms if vol[g] > 0.2 * max(vol.values())]
        front_u = max(along[g] for g in big)
        for g in geoms:
            if g not in big and along[g] > front_u - 0.005:
                self.part[g] = "handle"
            elif g in big and along[g] > front_u - 0.01:
                self.part[g] = "panel"
            else:
                self.part[g] = "floor" if _thin_axis_vertical(m, d, g) else "wall"

    def summary(self, m):
        out = {}
        for g, p in self.part.items():
            out.setdefault(p, []).append(m.geom(g).name or str(g))
        return out
