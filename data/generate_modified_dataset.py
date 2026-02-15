import json
import random
import copy

# ==========================================
# 1. タスク定義 (Problems)
# ==========================================
# 実験で使用するタスクと、理想的なオブジェクトリストの定義です。
# 参考: 以前の会話のJSONデータ
PROBLEMS = {
    "problem_1": {
        "description": "Set the table for two people (Normal)",
        "goal": "set the table for two people",
        "objects": ["table", "plate", "plate", "fork", "fork", "knife", "knife", "glass", "glass"],
        "scene": "office" # 推奨シーン
    },
    "problem_2": {
        "description": "Set the table for three people (Needs Recovery)",
        "goal": "set the table for three people",
        "objects": ["table", "plate", "plate", "plate", "fork", "fork", "fork", "knife", "knife", "knife", "bowl"],
        "scene": "office"
    },
    "problem_3": {
        "description": "Prepare a meeting room with refreshments",
        "goal": "serve water and snacks",
        "objects": ["table", "bottle_water", "bottle_water", "biscuits", "apple"],
        "scene": "office"
    },
    "problem_4": {
        "description": "Clean up the living room",
        "goal": "dispose of trash",
        "objects": ["banana_peel", "cola_can", "rubbish_bin"],
        "scene": "kemblesville"
    }
}

# ==========================================
# 2. 障害生成の設定 (Perturbations)
# ==========================================
# 発生させる障害のパターン定義
PERTURBATION_RULES = {
    "dirty": {
        "locations": ["sink", "trash_bin", "floor", "kitchen_bench"],
        "states": ["dirty", "stained"],
    },
    "hidden": {
        "locations": ["cupboard", "fridge", "drawer", "cabinet"],
        "states": ["closed", "inside"], # 親コンテナの状態
    },
    "misplaced": {
        "locations": ["floor", "under_sofa", "wrong_desk"],
        "states": ["free"],
    }
}

# 障害が発生する確率 (0.0 ~ 1.0)
PERTURBATION_PROBABILITY = 0.4

def generate_perturbed_graph(object_list):
    """
    オブジェクト名のリストから、障害（汚れや隠蔽）を含んだシーングラフの差分リストを生成する
    """
    current_scene_objects = []
    
    # 同じ名前のオブジェクトを区別するためのカウンタ (plate_1, plate_2...)
    obj_counts = {}

    for obj_name in object_list:
        # ID生成
        if obj_name not in obj_counts:
            obj_counts[obj_name] = 1
        else:
            obj_counts[obj_name] += 1
        
        # 例: plate -> plate_1
        # 注: delta.py側では startswith(label) でマッチングさせるため、
        # ここでのIDは一意であればOKです。
        obj_id = f"{obj_name}_{obj_counts[obj_name]}"
        
        # 確率判定: このオブジェクトに障害を発生させるか？
        # (テーブルやゴミ箱などの大型家具は除外する)
        is_furniture = obj_name in ["table", "rubbish_bin", "desk", "chair"]
        
        if not is_furniture and random.random() < PERTURBATION_PROBABILITY:
            # 障害の種類をランダム選択
            p_type = random.choice(["dirty", "hidden", "misplaced"])
            rule = PERTURBATION_RULES[p_type]
            
            # ロケーションと状態を決定
            new_loc = random.choice(rule["locations"])
            new_state = random.choice(rule["states"])
            
            perturbation = {
                "id": obj_id,
                "label": obj_name, # マッチング用ラベル
                "location": new_loc,
                "state": new_state,
                "perturbation_type": p_type
            }
            current_scene_objects.append(perturbation)
    
    return current_scene_objects

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print("Generating modified_scene_graphs.json ...")
    
    output_data = {"problems": {}}

    for p_id, p_info in PROBLEMS.items():
        print(f"  Processing {p_id}: {p_info['description']}")
        
        # 元の情報をコピー
        problem_entry = copy.deepcopy(p_info)
        
        # 障害リストを生成
        mods = generate_perturbed_graph(p_info["objects"])
        
        # もし運悪く障害がゼロだった場合、最後のアイテムを強制的に汚す（実験用）
        if not mods and len(p_info["objects"]) > 1:
            target = p_info["objects"][-1]
            if target not in ["table"]:
                mods.append({
                    "id": f"{target}_99", 
                    "label": target, 
                    "state": "dirty", 
                    "location": "sink",
                    "perturbation_type": "forced_dirty"
                })
        
        problem_entry["current_scene_graph"] = mods
        output_data["problems"][p_id] = problem_entry
        
        # ログ表示
        for m in mods:
            print(f"    -> {m['label']} is {m['state']} at {m['location']} ({m['perturbation_type']})")

    # ファイル保存
    filename = "modified_scene_graphs.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSuccessfully saved to {filename}")

if __name__ == "__main__":
    main()