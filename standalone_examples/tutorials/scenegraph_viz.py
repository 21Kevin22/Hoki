import json
import matplotlib
matplotlib.use('Agg') # 画面のない環境でも動作するように設定
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
import os

# 1. 提供されたJSONデータ
raw_json_input = """
{
    "frame": 12,
    "timestamp": 0.2,
    "nodes": [
        {
            "id": "unknown_0",
            "label": "bottle",
            "properties": {
                "position_3d": [0.0, 0.0, 0.0],
                "affordance": ["graspable", "liftable", "pourable", "containable"],
                "bbox_2d": [420, 474, 442, 572]
            }
        },
        {
            "id": "unknown_0",
            "label": "bottle",
            "properties": {
                "position_3d": [0.0, 0.0, 0.0],
                "affordance": ["graspable", "liftable", "pourable", "containable"],
                "bbox_2d": [461, 468, 482, 565]
            }
        },
        {
            "id": "unknown_0",
            "label": "bottle",
            "properties": {
                "position_3d": [0.0, 0.0, 0.0],
                "affordance": ["graspable", "liftable", "pourable", "containable"],
                "bbox_2d": [502, 463, 521, 559]
            }
        },
        {
            "id": "unknown_0",
            "label": "bottle",
            "properties": {
                "position_3d": [0.0, 0.0, 0.0],
                "affordance": ["graspable", "liftable", "pourable", "containable"],
                "bbox_2d": [540, 457, 561, 553]
            }
        },
        {
            "id": "unknown_0",
            "label": "bottle",
            "properties": {
                "position_3d": [0.0, 0.0, 0.0],
                "affordance": ["graspable", "liftable", "pourable", "containable"],
                "bbox_2d": [578, 452, 599, 547]
            }
        }
    ]
}
"""

def build_hierarchy_from_flat_json(json_str):
    """
    フラットな検出データを、論文のような階層構造データに変換します。
    Building -> Floor -> Room -> Items -> Affordances
    """
    data = json.loads(json_str)
    nodes = data.get("nodes", [])

    # アイテム層の構築
    item_children = []
    for i, node in enumerate(nodes):
        # IDが重複しているため、インデックスを使ってユニークにする
        label = node.get("label", "item")
        unique_name = f"{label}_{i}"
        
        affordances = node["properties"].get("affordance", [])
        
        item_children.append({
            "name": unique_name,
            "type": "item",
            "affordances": affordances
        })

    # 階層構造の組み立て
    hierarchy = {
        "name": "Simulation_World",
        "type": "root",
        "children": [
            {
                "name": "Lab_Floor",
                "type": "floor",
                "children": [
                    {
                        "name": "Table_Area",
                        "type": "room",
                        "children": item_children # ここに解析したアイテムを格納
                    }
                ]
            }
        ]
    }
    return hierarchy

def visualize_3d_hierarchical_scene_graph(scene_data):
    G = nx.DiGraph()
    pos_3d = {} 
    
    # 論文やDELTAの図に近い配色定義
    color_map = {
        "root": "#FF9999",       # 赤 (Root)
        "floor": "#FFCC99",      # オレンジ (Floor)
        "room": "#99FF99",       # 緑 (Room)
        "item": "#99CCFF",       # 青 (Item)
        "affordance": "#FFFF99"  # 黄 (Affordance)
    }

    # 再帰的にノード配置を計算する関数
    def add_nodes_3d(node, parent_name=None, x=0, y=0, z=0, width=1.0):
        name = node["name"]
        node_type = node["type"]
        
        # グラフへの追加
        G.add_node(name, type=node_type)
        pos_3d[name] = (x, y, z)
        
        if parent_name:
            G.add_edge(parent_name, name)

        # 子供要素の取得
        children = node.get("children", [])
        affordances = node.get("affordances", [])
        
        # 配置計算用の全要素数
        total_elements = len(children) + len(affordances)
        
        if total_elements > 0:
            step = width / total_elements
            start_x = x - (width / 2) + (step / 2)
            current_idx = 0
            
            # 通常の子ノード (再帰処理)
            for child in children:
                child_x = start_x + (current_idx * step)
                # 深くなるにつれてZを減らす。Yを少し散らして見やすくする
                child_y = y + (np.random.rand() - 0.5) * 0.5
                add_nodes_3d(child, name, child_x, child_y, z - 1, width=width/1.5)
                current_idx += 1
            
            # アフォーダンスノード (末端)
            for aff in affordances:
                # 名前を一意にする
                aff_unique_name = f"{name}_{aff}"
                G.add_node(aff_unique_name, type="affordance")
                
                aff_x = start_x + (current_idx * step)
                # アフォーダンスは少し手前(Y)に配置
                pos_3d[aff_unique_name] = (aff_x, y - 0.5, z - 1) 
                G.add_edge(name, aff_unique_name)
                current_idx += 1

    # 構築開始 (Z=4から)
    add_nodes_3d(scene_data, z=4, width=12.0)

    # 描画設定
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f0f0f0') # 背景色

    # エッジ描画
    for edge in G.edges():
        p1 = pos_3d[edge[0]]
        p2 = pos_3d[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                c='gray', alpha=0.4, linewidth=0.8)

    # ノード描画
    for node in G.nodes():
        x, y, z = pos_3d[node]
        n_type = nx.get_node_attributes(G, 'type')[node]
        c = color_map.get(n_type, 'gray')
        
        # サイズ調整
        s = 50 if n_type == 'affordance' else 250
        if n_type == 'root': s = 500
            
        ax.scatter(x, y, z, c=c, s=s, edgecolors='black', alpha=0.8)
        
        # ラベル表示
        if n_type == 'affordance':
            # アフォーダンス名はID部分を取り除いて表示
            label_text = node.split('_')[-1]
            ax.text(x, y, z-0.2, label_text, fontsize=7, ha='center', color='#333333')
        else:
            ax.text(x, y, z+0.2, node, fontsize=9, ha='center', weight='bold')

    # 軸を消してグラフのみ表示
    ax.set_axis_off()
    
    # 見やすい角度に調整
    ax.view_init(elev=25, azim=10)
    
    plt.title("Generated 3D Scene Graph from JSON Data", fontsize=16)
    
    output_file = "json_3d_scenegraph.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"3D Scene Graph saved to: {output_file}")

if __name__ == "__main__":
    # JSON変換
    hierarchical_data = build_hierarchy_from_flat_json(raw_json_input)
    # 描画実行
    visualize_3d_hierarchical_scene_graph(hierarchical_data)