import subprocess
import os
import sys
import shutil
import time

# 画像が保存されるフォルダ
image_dir = os.path.join(os.getcwd(), "pddl_final_results", "rgb")

# 出力する動画パス
output_video_path = os.path.join(os.getcwd(), "pddl_final_results", "robot_fill_complete.mp4")

# シミュレーション実行スクリプトのパス
simulation_script = os.path.join(os.getcwd(), "robot_plan.py") 
python_cmd = "./python.sh" 

fps = 30 

# OpenCVのインポートチェック
try:
    import cv2
except ImportError:
    print("❌ OpenCV (cv2) がインストールされていません。")
    print("   ./python.sh -m pip install opencv-python を実行してください。")
    sys.exit(1)

# =========================================================
# 関数定義
# =========================================================

def validate_images(folder_path):
    """
    フォルダ内の画像をチェックし、有効な画像リストを返す。
    壊れている画像が見つかった場合は報告する。
    """
    if not os.path.exists(folder_path):
        return [], 0 # フォルダがない

    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    if not files:
        return [], 0 # 画像がない

    valid_images = []
    broken_count = 0

    print(f"🔍 画像の健全性をチェック中 ({len(files)}枚)...")

    for i, filename in enumerate(files):
        path = os.path.join(folder_path, filename)
        
        # 1. ファイルサイズチェック
        if os.path.getsize(path) == 0:
            print(f"   ⚠️ サイズ0のファイルを検出: {filename}")
            broken_count += 1
            continue

        # 2. OpenCVによる読み込みチェック
        img = cv2.imread(path)
        if img is None:
            print(f"   ⚠️ 破損した画像データを検出: {filename}")
            broken_count += 1
            continue
        
        valid_images.append(filename)

    return valid_images, broken_count

def run_simulation_clean():
    """既存の画像フォルダを掃除してからシミュレーションを実行する"""
    print("🔄 シミュレーションを実行して画像を再生成します...")
    
    if os.path.exists(image_dir):
        try:
            print(f"🗑️ 古い画像フォルダを削除中: {image_dir}")
            shutil.rmtree(image_dir)
        except Exception as e:
            print(f"⚠️ 削除警告: {e}")

    if not os.path.exists(simulation_script):
        print(f"❌ エラー: スクリプトが見つかりません: {simulation_script}")
        return False

    try:
        cmd = [python_cmd, simulation_script]
        if os.path.exists(python_cmd) and not os.access(python_cmd, os.X_OK):
             os.chmod(python_cmd, 0o755)

        print(f"🚀 コマンド実行: {' '.join(cmd)}")
        # 実行完了を待つ
        ret = subprocess.call(cmd)
        
        if ret != 0:
            print("❌ シミュレーション実行中にエラーが発生しました。")
            return False
            
        return True
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        return False

def create_video(valid_files, input_folder, output_path, fps=30):
    """有効な画像リストから動画を作成する"""
    if not valid_files:
        print("❌ 動画にするための有効な画像がありません。")
        return

    print(f"🎥 {len(valid_files)} 枚の有効な画像から動画を作成しています...")

    # 最初の画像からサイズを取得
    first_path = os.path.join(input_folder, valid_files[0])
    frame = cv2.imread(first_path)
    if frame is None:
        print("❌ 最初の画像の読み込みに失敗しました。")
        return
        
    height, width, layers = frame.shape

    # 動画ライターの準備
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    count = 0
    for filename in valid_files:
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)
        
        if img is not None:
            # 万が一サイズが違う画像が混じっていた場合のリサイズ（念のため）
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            video.write(img)
            count += 1
        
        if count % 100 == 0:
            print(f"   ... {count}/{len(valid_files)} フレーム処理中")

    cv2.destroyAllWindows()
    video.release()
    print(f"✅ 動画の作成が完了しました！: {output_path}")

# =========================================================
# メイン処理フロー
# =========================================================
if __name__ == "__main__":
    # 1. 現状のチェック
    valid_files, broken_count = validate_images(image_dir)
    
    need_rerun = False

    if len(valid_files) == 0:
        print("⚠️ 画像が見つかりません。")
        need_rerun = True
    elif broken_count > 0:
        print(f"⚠️ {broken_count} 枚の破損画像を検出しました。完全な動画を作るために再実行します。")
        need_rerun = True
    elif len(valid_files) < 10: # 極端に枚数が少ない場合も怪しいので再実行
        print(f"⚠️ 画像枚数が少なすぎます ({len(valid_files)}枚)。正常に終了していない可能性があるため再実行します。")
        need_rerun = True
    else:
        print("✅ 画像データは健全です。")

    # 2. 必要ならシミュレーション再実行
    if need_rerun:
        success = run_simulation_clean()
        if not success:
            print("❌ シミュレーションの再実行に失敗しました。処理を中断します。")
            sys.exit(1)
        
        # 再実行後に再チェック
        valid_files, broken_count = validate_images(image_dir)
        if not valid_files:
            print("❌ 再実行しましたが、画像が生成されませんでした。コードを確認してください。")
            sys.exit(1)

    # 3. 動画作成
    create_video(valid_files, image_dir, output_video_path, fps)