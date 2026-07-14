import copy

def flatten_scene_graph(sg):
    """
    シーングラフをフラットな辞書に変換し、比較しやすくするヘルパー関数。
    """
    flat_data = {}

    # 1. Agent & Human (トップレベル情報)
    if "agent" in sg:
        pos = sg["agent"].get("position")
        state = sg["agent"].get("state")
        flat_data["AGENT"] = {
            "room": pos, 
            "state": state,
            "parent": None
        }
    if "human" in sg:
        flat_data[f"HUMAN_{sg['human']['name']}"] = {
            "room": sg["human"].get("position"),
            "state": sg["human"].get("state"),
            "parent": None
        }

    # 2. Rooms traverse
    if "rooms" in sg:
        for room_name, room_data in sg["rooms"].items():
            # A. Direct Items (Allensville style)
            if "items" in room_data:
                for item_id, item_data in room_data["items"].items():
                    flat_data[item_id] = {
                        "room": room_name,
                        "parent": None, # 部屋に直置き
                        "state": item_data.get("state", "unknown"),
                        "accessible": item_data.get("accessible", True)
                    }

            # B. Assets and Items inside Assets (Office style)
            if "assets" in room_data:
                for asset_id, asset_data in room_data["assets"].items():
                    # Asset自体 (Desk, Cabinet etc.)
                    flat_data[asset_id] = {
                        "room": room_name,
                        "parent": None,
                        "state": asset_data.get("state", "unknown"),
                        "accessible": asset_data.get("accessible", True)
                    }
                    
                    # Items inside the Asset
                    if "items" in asset_data:
                        for item_id, item_data in asset_data["items"].items():
                            flat_data[item_id] = {
                                "room": room_name,
                                "parent": asset_id, # 家具の中にある
                                "state": item_data.get("state", "unknown"),
                                "accessible": item_data.get("accessible", True),
                                "relation": item_data.get("relation", "in") # in or on
                            }
    return flat_data

def compare_scene_graphs(ideal_sg, current_sg):
    """
    理想(ideal)と現実(current)を比較し、差分レポートを返す
    """
    ideal_flat = flatten_scene_graph(ideal_sg)
    current_flat = flatten_scene_graph(current_sg)

    diff_report = {
        "state_changes": [],      # 状態変化 (Dirty, Closed...)
        "location_changes": [],   # 場所移動 (Different Room, Inside different obj)
        "missing_objects": [],    # なくなった物体
        "new_objects": []         # 未知の物体 (障害物)
    }

    # 1. Check Ideal objects (Missing, Moved, Changed)
    for obj_id, ideal_props in ideal_flat.items():
        if obj_id not in current_flat:
            diff_report["missing_objects"].append(obj_id)
            continue
        
        curr_props = current_flat[obj_id]

        # Check Location (Room or Parent Container)
        if (ideal_props["room"] != curr_props["room"]) or (ideal_props["parent"] != curr_props["parent"]):
            diff_report["location_changes"].append({
                "object": obj_id,
                "from": {"room": ideal_props["room"], "parent": ideal_props["parent"]},
                "to": {"room": curr_props["room"], "parent": curr_props["parent"]}
            })
        
        # Check State (Closed/Open, Clean/Dirty, etc.)
        if ideal_props["state"] != curr_props["state"]:
            diff_report["state_changes"].append({
                "object": obj_id,
                "expected": ideal_props["state"],
                "actual": curr_props["state"]
            })

    # 2. Check New objects
    for obj_id in current_flat:
        if obj_id not in ideal_flat:
            diff_report["new_objects"].append({
                "object": obj_id,
                "location": current_flat[obj_id]["room"],
                "parent": current_flat[obj_id]["parent"],
                "state": current_flat[obj_id]["state"]
            })

    return diff_report