# next_hidden — 閉じた電子レンジの中のマグを取り出す探索実験（凍結 π0.5）

本線の外の探索実験。手順・決まりは `PREREG_next.md`、結果は `RESULTS_next.md`（日付つき追記）。

| ファイル | 役割 |
|---|---|
| `scripts/env_check.py` | 手順 4：GPU・本体・描画・重みの有無を調べ `results/env_report.json` に保存 |
| `scripts/monitor.sh` | 手順 5：1 分ごとに nvidia-smi と本体メモリを記録。95% 超で `STOP_NEW_EPISODES` を作る。30 分エピソードが終わらなければ警告 |
| `scripts/make_manifest.py` | 手順 3/7：ファイルの sha256、git コミット、pip freeze を MANIFEST に記録 |
| `task/` | 持ってくるタスク（BDDL、初期状態、確認スクリプト）。未着 |

```bash
source /home/ubuntu/slocal1/third_party/stowsafe_env.sh   # LIBERO + MuJoCo(EGL)
python scripts/env_check.py
bash scripts/monitor.sh results/monitor.log &
python scripts/make_manifest.py task/ scripts/ --out MANIFEST.json
```

決まり（抜粋）：1 プロセス 1 GPU（`CUDA_VISIBLE_DEVICES`）、π0.5 は `XLA_PYTHON_CLIENT_PREALLOCATE=false`、
Qwen と π0.5 を同じ時間に同じ GPU に載せない、まず 1 枚・1 本で最大メモリと時間を測る。
