import json
from pathlib import Path

import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.patheffects as patheffects


def _hierarchy_pos(nodes):
    layer_nodes = {}
    for n in nodes:
        layer = 0 if n.get("type") == "mug" else 1
        layer_nodes.setdefault(layer, []).append(n["id"])

    pos = {}
    y_step = 2.5
    for layer, ids in layer_nodes.items():
        n = len(ids)
        if n == 1:
            xs = [0.0]
        else:
            xs = [i - (n - 1) / 2.0 for i in range(n)]
        for x, node_id in zip(xs, ids):
            pos[node_id] = (x, -layer * y_step)
    return pos, layer_nodes


def _find_latest_sim_image(json_path: Path) -> Path:
    base = json_path.parent
    candidates = [
        base / "frames" / "main",
        base / "frames",
        base / "rgb",
        base.parent / "frames" / "main",
        base.parent / "frames",
        base.parent / "rgb",
    ]
    for d in candidates:
        if d.exists():
            imgs = sorted(d.glob("rgb_*.png"))
            if imgs:
                return imgs[-1]
    return Path("")


def _layer_map(layer_nodes):
    out = {}
    for layer, ids in layer_nodes.items():
        for nid in ids:
            out[nid] = layer
    return out


def _normalize_scene_graph_payload(data):
    if not isinstance(data, dict):
        return [], []

    if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        return data.get("nodes", []), data.get("edges", [])

    sg = data.get("scene_graph") if isinstance(data.get("scene_graph"), dict) else data
    rooms = sg.get("rooms", {}) if isinstance(sg, dict) else {}

    nodes = []
    edges = []

    # Room nodes and item containment edges
    for room_name, room_data in rooms.items():
        nodes.append({"id": room_name, "type": "room"})
        items = room_data.get("items", {}) if isinstance(room_data, dict) else {}
        for item_name, item_data in items.items():
            node_type = "mug" if item_name.startswith("mug") else "object"
            node_payload = {"id": item_name, "type": node_type}
            if isinstance(item_data, dict) and "affordance" in item_data:
                aff_raw = item_data.get("affordance", [])
                if isinstance(aff_raw, list):
                    node_payload["affordance"] = {k: True for k in aff_raw}
            nodes.append(node_payload)
            edges.append({"from": room_name, "to": item_name, "relation": "contains"})

        neighbors = room_data.get("neighbor", []) if isinstance(room_data, dict) else []
        if isinstance(neighbors, list):
            for nb in neighbors:
                if isinstance(nb, str):
                    edges.append({"from": room_name, "to": nb, "relation": "neighbor"})

    # Agent node
    agent = sg.get("agent", {}) if isinstance(sg, dict) else {}
    if isinstance(agent, dict):
        nodes.append({"id": "agent", "type": "agent"})
        agent_pos = agent.get("position")
        if isinstance(agent_pos, str) and agent_pos:
            edges.append({"from": "agent", "to": agent_pos, "relation": "in"})

    # Detection relations (optional)
    detection = data.get("detection", {})
    relations = detection.get("relations", []) if isinstance(detection, dict) else []
    if isinstance(relations, list):
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            s = rel.get("subject")
            r = rel.get("relation")
            o = rel.get("object")
            if isinstance(s, str) and isinstance(r, str) and isinstance(o, str):
                edges.append({"from": s, "to": o, "relation": r})

    # de-dup by id while preserving order
    uniq_nodes = []
    seen = set()
    for n in nodes:
        nid = n.get("id")
        if not nid or nid in seen:
            continue
        seen.add(nid)
        uniq_nodes.append(n)

    return uniq_nodes, edges


def render_simple_scene_graph(json_path: str, sim_image_path: str = "") -> str:
    """Render a scene graph PNG next to the JSON file.
    Includes an optional simulation snapshot for correspondence."""
    json_path = Path(json_path)
    if not json_path.exists():
        return ""
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return ""

    nodes, edges = _normalize_scene_graph_payload(data)

    G = nx.DiGraph()
    for n in nodes:
        node_id = n.get("id", "")
        if not node_id:
            continue
        label = f"{node_id}\n({n.get('type', 'node')})"
        afford = n.get("affordance")
        if isinstance(afford, dict):
            aff_keys = [k for k, v in afford.items() if v]
            if aff_keys:
                label += "\n" + ",".join(aff_keys)
        G.add_node(node_id, label=label, ntype=n.get("type", "node"))
    for e in edges:
        G.add_edge(e.get("from"), e.get("to"), label=e.get("relation", ""))

    pos = None
    layer_nodes = None
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except Exception:
        pos, layer_nodes = _hierarchy_pos(nodes)

    if layer_nodes is None:
        layer_nodes = {}
        for node_id, (_, y) in pos.items():
            layer = 0 if y >= 0 else 1
            layer_nodes.setdefault(layer, []).append(node_id)

    layer_map = _layer_map(layer_nodes)

    sim_img_path = Path(sim_image_path) if sim_image_path else _find_latest_sim_image(json_path)

    if sim_img_path.exists():
        fig = plt.figure(figsize=(8, 9))
        ax = fig.add_subplot(2, 1, 1)
        ax_img = fig.add_subplot(2, 1, 2)
        try:
            from PIL import Image
            img = Image.open(sim_img_path)
            ax_img.imshow(img)
            ax_img.set_title("Simulation View", fontsize=10)
            ax_img.axis("off")
        except Exception:
            ax_img.axis("off")
            ax_img.set_title("Simulation View", fontsize=10)
        # allow slight overlap between panels
        top_pos = ax.get_position()
        bot_pos = ax_img.get_position()
        ax.set_position([top_pos.x0, top_pos.y0 - 0.04, top_pos.width, top_pos.height + 0.04])
        ax_img.set_position([bot_pos.x0, bot_pos.y0, bot_pos.width, bot_pos.height + 0.02])
    else:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax_img = None

    # Camera-like background
    ax.set_facecolor("#f6f9ff")

    # soft gradient
    try:
        import numpy as np
        xmin = min(x for x, _ in pos.values()) - 1.2
        xmax = max(x for x, _ in pos.values()) + 1.2
        ymin = min(y for _, y in pos.values()) - 1.2
        ymax = max(y for _, y in pos.values()) + 1.2
        grad = np.linspace(1.0, 0.94, 256)
        grad = np.tile(grad, (256, 1))
        ax.imshow(grad, extent=[xmin, xmax, ymin, ymax], cmap="Blues", alpha=0.10, zorder=0)
        # faint grid
        for x in np.linspace(xmin, xmax, 8):
            ax.plot([x, x], [ymin, ymax], color="#e1ebf6", lw=0.6, zorder=0)
        for y in np.linspace(ymin, ymax, 6):
            ax.plot([xmin, xmax], [y, y], color="#e1ebf6", lw=0.6, zorder=0)
    except Exception:
        pass

    # Background panels by layer with shadow
    for layer, ids in layer_nodes.items():
        xs = [pos[n][0] for n in ids]
        ys = [pos[n][1] for n in ids]
        if not xs or not ys:
            continue
        pad_x = 0.9
        pad_y = 0.7
        x0, x1 = min(xs) - pad_x, max(xs) + pad_x
        y0, y1 = min(ys) - pad_y, max(ys) + pad_y
        color = "#e8f2ff" if layer == 0 else "#f2f7ff"
        shadow = FancyBboxPatch(
            (x0 + 0.05, y0 - 0.05), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0,
            facecolor="#000000",
            alpha=0.06,
            zorder=0,
        )
        ax.add_patch(shadow)
        panel = FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0.8,
            edgecolor="#c2d6ef",
            facecolor=color,
            alpha=0.38 if layer == 0 else 0.26,
            zorder=0.5,
        )
        ax.add_patch(panel)
        ax.text(x0 + 0.1, y1 - 0.2, f"Layer {layer}", fontsize=8, color="#5a7aa3", zorder=1)

    # styling by type and layer
    node_colors = []
    node_sizes = []
    node_alphas = []
    for n in G.nodes:
        ntype = G.nodes[n].get("ntype", "node")
        layer = layer_map.get(n, 0)
        if ntype == "mug":
            base_size = 1600
            color = "#b7d7ff"
        else:
            base_size = 1200
            color = "#cfe8ff"
        scale = max(0.65, 1.0 - 0.12 * layer)
        alpha = max(0.6, 0.95 - 0.15 * layer)
        node_colors.append(color)
        node_sizes.append(base_size * scale)
        node_alphas.append(alpha)

    # node shadows
    shadow_pos = {n: (pos[n][0] + 0.05, pos[n][1] - 0.05) for n in G.nodes}
    nx.draw_networkx_nodes(
        G, shadow_pos, ax=ax, node_color="#000000",
        node_size=[s * 1.02 for s in node_sizes], linewidths=0,
        alpha=0.18
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, edgecolors="#2f5f8f",
        node_size=node_sizes, linewidths=1.2, alpha=0.98
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrowstyle="->", arrowsize=14, edge_color="#4a4a4a",
        width=1.6, alpha=0.7
    )

    # labels with badges + shadow
    labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
    for node_id, (x, y) in pos.items():
        txt = ax.text(
            x, y, labels.get(node_id, node_id),
            ha="center", va="center", fontsize=8, zorder=4,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#7aa0c8", alpha=0.92)
        )
        txt.set_path_effects([
            patheffects.withSimplePatchShadow(offset=(1, -1), shadow_rgbFace=(0, 0, 0, 0.25))
        ])

    # edge labels
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    # correspondence cue: arrow from sim view to root node
    if ax_img is not None and G.nodes:
        root = None
        for node_id in G.nodes:
            if G.nodes[node_id].get("ntype") == "mug":
                root = node_id
                break
        if root is None:
            root = list(G.nodes)[0]
        xy = pos.get(root, (0, 0))
        con = ConnectionPatch(
            xyA=(0.5, 0.98), coordsA=ax_img.transAxes,
            xyB=xy, coordsB=ax.transData,
            arrowstyle="->", lw=1.5, color="#d25555", alpha=0.75
        )
        fig.add_artist(con)
        ax.scatter([xy[0]], [xy[1]], s=2000, facecolors="none", edgecolors="#d25555", linewidths=1.2, zorder=4)

    ax.set_title("3D Scene Graph (Hierarchical)", fontsize=10)
    ax.axis("off")

    png_path = json_path.with_suffix(".png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close()
    return str(png_path)


if __name__ == "__main__":
    import sys

    default_json = "/home/ubuntu/slocal/evaluation/vlm_delta_original/scene_graph_json/detected_scene_graph_1774027842832.json"
    json_input = sys.argv[1] if len(sys.argv) > 1 else default_json
    sim_path = sys.argv[2] if len(sys.argv) > 2 else ""

    out = render_simple_scene_graph(json_input, sim_path)
    if out:
        print(out)
    else:
        raise SystemExit(f"Failed to render scene graph from: {json_input}")
