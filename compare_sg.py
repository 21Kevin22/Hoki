import json
import os
import copy

def compare_nodes_by_label_and_count(ideal_data, real_data):
    """
    3D座標が使えない場合のために、ラベルと出現順序で比較を行う関数
    """
    # データの正規化 (dict -> list)
    ideal_nodes = ideal_data.get("nodes", []) if isinstance(ideal_data, dict) else ideal_data
    real_nodes = real_data.get("nodes", []) if isinstance(real_data, dict) else real_data

    if isinstance(real_nodes, dict) and "objects" in real_nodes: real_nodes = real_nodes["objects"]
    if isinstance(ideal_nodes, dict) and "objects" in ideal_nodes: ideal_nodes = ideal_nodes["objects"]

    # マッチング用コピー
    real_nodes_working = copy.deepcopy(real_nodes)
    
    report = {
        "summary": {"total_ideal": len(ideal_nodes), "total_real": len(real_nodes)},
        "missing_objects": [],
        "new_objects": [],
        "matched_objects": []
    }

    # --- A. Ideal を基準に Real を探す ---
    for i, ideal_obj in enumerate(ideal_nodes):
        target_label = ideal_obj.get("label", "unknown")
        
        # 3D座標があれば取得、なければ None
        if "properties" in ideal_obj:
            ideal_pos = ideal_obj["properties"].get("position_3d")
        else:
            ideal_pos = ideal_obj.get("position_3d")

        # Realの中から「同じラベル」で「まだマッチしていない」ものを探す
        match = None
        for real_obj in real_nodes_working:
            if real_obj.get("_matched", False):
                continue
            
            if real_obj.get("label") == target_label:
                match = real_obj
                break
        
        if match:
            # マッチした場合
            match["_matched"] = True
            
            # Real側の座標やBBoxを取得
            real_pos = match.get("properties", {}).get("position_3d", match.get("position_3d", [0,0,0]))
            bbox = match.get("properties", {}).get("bbox_2d", match.get("bbox_2d"))

            report["matched_objects"].append({
                "label": target_label,
                "ideal_id": f"ideal_{i}",
                "ideal_pos": ideal_pos,
                "real_pos": real_pos, # [0,0,0] でもそのまま表示
                "bbox_2d": bbox,
                "status": "Match Found (by Label)"
            })
        else:
            # 見つからない場合
            report["missing_objects"].append({
                "label": target_label,
                "expected_pos": ideal_pos
            })

    # --- B. マッチしなかった Real を探す ---
    for r_obj in real_nodes_working:
        if not r_obj.get("_matched", False):
            pos = r_obj.get("properties", {}).get("position_3d", r_obj.get("position_3d"))
            bbox = r_obj.get("properties", {}).get("bbox_2d", r_obj.get("bbox_2d"))

            report["new_objects"].append({
                "label": r_obj.get("label"),
                "found_pos": pos,
                "bbox_2d": bbox
            })

    return report

# =========================================================
# メイン処理
# =========================================================
if __name__ == "__main__":
    FILE_IDEAL = "ideal_scenegraph.json"
    FILE_REAL  = "real_scenegraph.json"

    print(f"📂 ファイル読み込み中...")
    print(f"   Target 1 (Ideal): {FILE_IDEAL}")
    print(f"   Target 2 (Real) : {FILE_REAL}")

    if not os.path.exists(FILE_IDEAL) or not os.path.exists(FILE_REAL):
        print("❌ エラー: ファイルが見つかりません。")
        exit()

    try:
        with open(FILE_IDEAL, 'r') as f:
            data_ideal = json.load(f)
        
        with open(FILE_REAL, 'r') as f:
            data_real = json.load(f)

        print("✅ 読み込み成功。ラベルベースでの比較を開始します...")

        # 比較実行
        result = compare_nodes_by_label_and_count(data_ideal, data_real)

        # 結果表示
        print("\n=== 📊 比較レポート (ラベルマッチング) ===")
        print(json.dumps(result, indent=4, ensure_ascii=False))

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")