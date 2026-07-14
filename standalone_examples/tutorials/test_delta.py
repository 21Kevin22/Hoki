import json
import os
import sys

# 書き込み先のパス (メインスクリプトと同じ場所を想定)
# 環境に合わせて調整してください
OUTPUT_DIR = os.getcwd() + "/bottle_sorting_output"
JSON_PATH = os.path.join(OUTPUT_DIR, "actions.json")

print(f"[MockDelta] Updating plan at: {JSON_PATH}")

# テスト用の新しいプラン（順番を逆にするなど、変化がわかるようにする）
new_plan_data = {
    "actions": [
        "(pick bottle_2)", "(place)",  # 青から始める
        "(pick bottle_1)", "(place)",  # 緑
        "(pick bottle_0)", "(place)"   # 赤
    ]
}

# ディレクトリがない場合は作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# JSONファイルに書き込み
with open(JSON_PATH, "w") as f:
    json.dump(new_plan_data, f, indent=2)

print("[MockDelta] actions.json has been updated!")