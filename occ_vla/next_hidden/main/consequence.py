"""CQ: consequence-gated safety (PREREG_main.md 2026-11-23 (5)), stage 1 = upper bound with the TRUE simulator as twin.
Before executing a command, the next H commands (current + rest of the plan) are tried in the twin from the current state.
Harm = some bystander (movable object that is not a target / destination / the task container) moves > 1 mm
(SafeLIBERO: sum over axes of |displacement|). Only harmful commands are changed; contact itself is never a reason."""
import numpy as np
from collisions import CollisionRecorder
from parts import TaskParts

H = 8
THR = 0.001


def sync_internal(src_env, dst_env):
    """Copy the robosuite state that is NOT in the MuJoCo state: the gripper's speed-limited command and the OSC goal."""
    ds, dd = src_env.sim.data._data, dst_env.sim.data._data
    dd.qacc_warmstart[:] = ds.qacc_warmstart; dd.ctrl[:] = ds.ctrl; dd.qfrc_applied[:] = ds.qfrc_applied
    rs, rd = src_env.env.robots[0], dst_env.env.robots[0]
    rd.gripper.current_action = np.array(rs.gripper.current_action, copy=True)
    for k in ("goal_pos", "goal_ori", "relative_ori", "ori_ref"):
        v = getattr(rs.controller, k, None)
        setattr(rd.controller, k, None if v is None else np.array(v, copy=True))


def warm_up(env, twin):
    twin.env.timestep = 0; twin.env.done = False
    twin.sim.set_state_from_flattened(env.sim.get_state().flatten()); twin.sim.forward()
    sync_internal(env, twin)
    twin.step([0.0] * 6 + [-1.0])


def _perp(v):
    """Two horizontal directions perpendicular to the translation v, and up."""
    h = np.array([v[0], v[1], 0.0])
    if np.linalg.norm(h) < 1e-6:
        h = np.array([1.0, 0.0, 0.0])
    h /= np.linalg.norm(h)
    side = np.array([-h[1], h[0], 0.0])
    return [side, -side, np.array([0.0, 0.0, 1.0])]


class ConsequenceFilter:
    def __init__(self, twin, task, env=None):
        self.twin, self.task, self.env = twin, task, env
        m, d = twin.sim.model._model, twin.sim.data._data
        self.byst = CollisionRecorder(TaskParts(task, m, d), m, d).bystanders
        self.stats = dict(checks=0, harm_predicted=0, corrected=0, none_safe=0, rollouts=0)
        if env is not None:                    # warm-up: the first step after a twin reset differs (pre-run check)
            warm_up(env, twin)

    def _harm(self, state, seq):
        tw = self.twin
        tw.env.timestep = 0; tw.env.done = False
        tw.sim.set_state_from_flattened(state); tw.sim.forward()
        if self.env is not None:
            sync_internal(self.env, tw)
        d = tw.sim.data._data
        p0 = {b: d.xpos[b].copy() for b in self.byst}
        worst = 0.0
        for a in seq[:H]:
            tw.step(np.clip(np.asarray(a, float), -1, 1).tolist())
            worst = max([worst] + [float(np.abs(d.xpos[b] - p0[b]).sum()) for b in self.byst])
        self.stats["rollouts"] += 1
        return worst

    def predict(self, state, u, plan):
        """Bystander bodies the UNCHANGED commands would move > THR within H steps (for CQB's fences)."""
        tw = self.twin
        tw.env.timestep = 0; tw.env.done = False
        tw.sim.set_state_from_flattened(state); tw.sim.forward()
        if self.env is not None:
            sync_internal(self.env, tw)
        d = tw.sim.data._data
        p0 = {b: d.xpos[b].copy() for b in self.byst}
        moved = set()
        seq = [np.asarray(u, float)] + [np.asarray(p, float) for p in plan]
        seq += [seq[-1]] * (H - len(seq))          # the plan holds <= 4 more steps: hold the last command up to H
        for a in seq[:H]:
            tw.step(np.clip(a, -1, 1).tolist())
            moved |= {b for b in self.byst if float(np.abs(d.xpos[b] - p0[b]).sum()) > THR}
        self.stats["checks"] += 1; self.stats["rollouts"] += 1
        if moved:
            self.stats["harm_predicted"] += 1
        return moved

    def act(self, state, u, plan):
        """Returns (action, info). info['replan'] = True when the command was changed."""
        self.stats["checks"] += 1
        u = np.asarray(u, float)
        seq = [u] + [np.asarray(p, float) for p in plan]
        if not self.byst:
            return u, dict(corrected=False, harm=0.0, replan=False)
        h0 = self._harm(state, seq)
        if h0 <= THR:
            return u, dict(corrected=False, harm=h0, replan=False)
        self.stats["harm_predicted"] += 1
        cands = []
        for s in (0.5, 0.25, 0.0):
            cands.append([np.concatenate([a[:3] * s, a[3:]]) for a in seq])
        for dvec in _perp(u[:3]):
            cands.append([np.concatenate([a[:3] + 0.5 * dvec, a[3:]]) for a in seq])
        scored = [(self._harm(state, c), float(np.linalg.norm(c[0][:3] - u[:3])), i, c) for i, c in enumerate(cands)]
        safe = [x for x in scored if x[0] <= THR]
        if safe:
            best = min(safe, key=lambda x: (x[1], x[2]))
        else:
            self.stats["none_safe"] += 1
            best = min(scored, key=lambda x: (x[0], x[1], x[2]))
        self.stats["corrected"] += 1
        a = np.clip(best[3][0], -1, 1)
        return a, dict(corrected=True, harm=h0, chosen=best[2], chosen_harm=best[0], replan=True)
