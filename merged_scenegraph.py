import json
import os

def merge_and_convert_format(source_data):
    """
    nodes (2D) と world_knowledge (3D) が分かれているデータを、
    指定された統合フォーマットに変換する関数
    """
    merged_data = {
        "frame": source_data.get("frame", 0),
        "timestamp": source_data.get("timestamp", 0.0),
        "nodes": []
    }

    # 1. データの取り出し
    # nodesがリストの場合と、辞書(count, objects)の場合に対応
    raw_nodes = source_data.get("nodes", [])
    if isinstance(raw_nodes, dict):
        visual_objects = raw_nodes.get("objects", [])
    else:
        visual_objects = raw_nodes

    # world_knowledgeも同様に対応
    raw_world = source_data.get("world_knowledge", {})
    if isinstance(raw_world, dict) and "objects" in raw_world:
        world_objects = raw_world.get("objects", [])
    else:
        world_objects = []

    # 2. マッチング処理用に視覚データをコピー（使ったら消すため）
    available_visuals = visual_objects.copy()

    # 3. World Knowledge (3D正解データ) をベースにループ
    for w_obj in world_objects:
        label = w_obj.get("label")
        
        # 新しいノード構造を作成
        new_node = {
            "id": w_obj.get("id"),
            "label": label,
            "properties": {
                "position_3d": w_obj.get("position_3d", [0,0,0]),
                "affordance": w_obj.get("affordance", [])
            }
        }

        # 対応する 2D BBox を探してマージする
        # (ラベルが一致する一番最初のものを使用)
        matched_visual = None
        for v_obj in available_visuals:
            if v_obj.get("label") == label:
                matched_visual = v_obj
                break
        
        if matched_visual:
            # 見つかったらBBoxを追加し、リストから削除（重複使用防止）
            new_node["properties"]["bbox_2d"] = matched_visual.get("bbox_2d", [])
            available_visuals.remove(matched_visual)
        else:
            # 見つからなかった場合（画面外など）
            new_node["properties"]["bbox_2d"] = None

        merged_data["nodes"].append(new_node)

    return merged_data

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
if __name__ == "__main__":
    # 変換したいファイル名
    INPUT_FILE = "real_scenegraph.json"  # 今持っているバラバラのデータ
    OUTPUT_FILE = "real_merged.json"     # 保存するきれいなデータ

    print(f"📂 読み込み中: {INPUT_FILE}")
    
    try:
        with open(INPUT_FILE, 'r') as f:
            source_data = json.load(f)

        # 変換実行
        result_data = merge_and_convert_format(source_data)

        # 保存
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(result_data, f, indent=4)
        
        print(f"✅ 変換成功！保存しました: {OUTPUT_FILE}")
        
        # 確認用表示
        print("\n--- 生成されたデータの先頭 ---")
        print(json.dumps(result_data["nodes"][0], indent=4))

    except FileNotFoundError:
        print(f"❌ ファイルが見つかりません: {INPUT_FILE}")
    except Exception as e:
        print(f"❌ エラー: {e}")