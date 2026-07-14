"""Minimal file-based RPC so the demo orchestrator (running in the
LIBERO/robosuite environment) can call pi0.5 (openpi's jax/torch==2.7.1
venv) and MMaDA-8B (its own torch==2.5.1/transformers==4.46.0 venv)
without merging three mutually-incompatible dependency pins into one
Python environment — see README "Setup" for why each needs its own venv.

Protocol: caller writes <id>.request.npz (+ .request.json for
non-array fields), then <id>.request.ready as a marker; worker polls
for *.request.ready, processes, writes <id>.response.npz(+.json) and
<id>.response.ready; caller polls for that and cleans up all four
files. Payloads are small (single images / short arrays) and MMaDA
calls are already ~50s, so filesystem polling overhead is negligible.
"""

import json
import time
import uuid
from pathlib import Path

import numpy as np

POLL_INTERVAL_S = 0.2


def call(rpc_dir: str, arrays: dict, fields: dict, timeout_s: float = 600) -> tuple[dict, dict]:
    """Client side: send a request, block until the response arrives."""
    rpc_dir = Path(rpc_dir)
    req_id = uuid.uuid4().hex
    np.savez(rpc_dir / f"{req_id}.request.npz", **arrays)
    (rpc_dir / f"{req_id}.request.json").write_text(json.dumps(fields))
    (rpc_dir / f"{req_id}.request.ready").touch()

    deadline = time.time() + timeout_s
    resp_ready = rpc_dir / f"{req_id}.response.ready"
    while not resp_ready.exists():
        if time.time() > deadline:
            raise TimeoutError(f"RPC call {req_id} in {rpc_dir} timed out after {timeout_s}s")
        time.sleep(POLL_INTERVAL_S)

    resp_arrays = dict(np.load(rpc_dir / f"{req_id}.response.npz"))
    resp_fields = json.loads((rpc_dir / f"{req_id}.response.json").read_text())
    for suffix in ("request.npz", "request.json", "request.ready", "response.npz", "response.json", "response.ready"):
        (rpc_dir / f"{req_id}.{suffix}").unlink(missing_ok=True)
    return resp_arrays, resp_fields


def serve(rpc_dir: str, handler) -> None:
    """Worker side: block forever, calling handler(arrays, fields) ->
    (resp_arrays, resp_fields) for each request."""
    rpc_dir = Path(rpc_dir)
    print(f"[rpc worker] serving {rpc_dir}", flush=True)
    while True:
        for ready_file in rpc_dir.glob("*.request.ready"):
            req_id = ready_file.name.removesuffix(".request.ready")
            arrays = dict(np.load(rpc_dir / f"{req_id}.request.npz"))
            fields = json.loads((rpc_dir / f"{req_id}.request.json").read_text())
            ready_file.unlink()
            (rpc_dir / f"{req_id}.request.npz").unlink(missing_ok=True)
            (rpc_dir / f"{req_id}.request.json").unlink(missing_ok=True)

            print(f"[rpc worker] handling {req_id}", flush=True)
            resp_arrays, resp_fields = handler(arrays, fields)

            np.savez(rpc_dir / f"{req_id}.response.npz", **resp_arrays)
            (rpc_dir / f"{req_id}.response.json").write_text(json.dumps(resp_fields))
            (rpc_dir / f"{req_id}.response.ready").touch()
            print(f"[rpc worker] done {req_id}", flush=True)
        time.sleep(POLL_INTERVAL_S)
