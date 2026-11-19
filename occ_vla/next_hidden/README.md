# next_hidden — 閉じた電子レンジの中のマグを取り出す探索実験（凍結 π0.5）

本線の外の探索実験。手順・決まりは `PREREG_next.md`、結果は `RESULTS_next.md`（日付つき追記）。

| ファイル | 役割 |
|---|---|
| `scripts/env_check.py` | 手順 4：GPU・本体・描画・重みの有無を調べ `results/env_report.json` に保存 |
| `scripts/monitor.sh` | 手順 5：1 分ごとに nvidia-smi と本体メモリを記録。95% 超で `STOP_NEW_EPISODES` を作る。30 分エピソードが終わらなければ警告 |
| `scripts/make_manifest.py` | 手順 3/7：ファイルの sha256、git コミット、pip freeze を MANIFEST に記録 |
| `bddl/`, `task/` | 作ったタスク（BDDL、`init_states.npz`） |
| `scripts/probe_scene6.py` | 元の libero_10 の電子レンジのタスクの扉の角度・内部の領域を測る |
| `scripts/build_init_states.py` | 6-1：マグを閉じた電子レンジに入れた初期状態を作り、全部の条件で確かめる |
| `scripts/verify_init_states.py` | 6-1：保存した状態を LIBERO の `set_init_state` で読み直して再確認 |
| `scripts/rgb_visibility.py` | 6-1：RGB にマグが写るか（マグを透明にした描画との差） |
| `scripts/scripted_expert.py` | 6-2：正解の位置を使う手本（開ける→取っ手をつまむ→出す→皿に置く）、扉の角度と入口の幅 |
| `scripts/run_gate_6_1_6_2.sh` | 6-1 と 6-2 を通しで実行 |
| `scripts/instruction_table.py` | 6-3：候補の指示文と学習の 40 タスクの対応 |

```bash
source /home/ubuntu/slocal1/third_party/stowsafe_env.sh   # LIBERO + MuJoCo(EGL)
python scripts/env_check.py
bash scripts/monitor.sh results/monitor.log &
RENDER_GPU=2 bash scripts/run_gate_6_1_6_2.sh      # 6-1 + 6-2（方策なし、描画のみ EGL）
python scripts/make_manifest.py task bddl scripts --out MANIFEST.json
```

決まり（抜粋）：1 プロセス 1 GPU（`CUDA_VISIBLE_DEVICES`）、π0.5 は `XLA_PYTHON_CLIENT_PREALLOCATE=false`、
Qwen と π0.5 を同じ時間に同じ GPU に載せない、まず 1 枚・1 本で最大メモリと時間を測る。
