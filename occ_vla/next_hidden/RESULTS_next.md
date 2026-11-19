# RESULTS_next — 日付つきで追記

## 2026-11-19 手順 4：環境の確認（GPU 不使用）
詳細は `results/env_report.json`。

| 項目 | 結果 |
|---|---|
| GPU | RTX A5000 ×3（compute 8.6、**Ampere**）、各 24 GB、空き ~24 GB、計算プロセスなし |
| ドライバ / CUDA | 595.84 / 13.2 |
| 共用 | ログイン中の他ユーザー・他ユーザーのプロセスなし（確認時点） |
| 本体メモリ | 物理 62.5 GiB、**cgroup 上限 52.0 GiB** |
| CPU / ディスク | 36 コア（cgroup 制限なし） / 空き 492 GiB |
| 型の方針 | Ampere なので bf16 でよい（FlashAttention 2 も使用可能な世代。ただし Qwen は sdpa で固定する予定） |
| 描画 | MuJoCo 3.0.0 + EGL で offscreen 描画 OK（`.venv_oft`, robosuite 1.4.1）。全実験 EGL に統一 |
| openpi / π0.5 | **未導入**。`pi05_libero` の重みもなし |
| Qwen2.5-VL-3B | **未取得** |

### 前のサーバーで落ちた件（手順 2）
前のサーバーのログはこのサーバーにないため、原因 (1)–(4) のどれかは **不明**。
なお候補 (3)（Turing の bf16/FA2）は、このサーバーが Ampere なので該当しない。

### 先に分かったこと（手順 6-3 の下調べ。タスク名一覧を見ただけ）
- 「電子レンジを開ける」は `libero_90` の `KITCHEN_SCENE7_open_the_microwave` のみ。
- 「電子レンジから取り出す」に当たるタスクは、LIBERO の全スイートに **ない**。
- 電子レンジのタスクは他に `libero_10` の `put the yellow and white mug in the microwave and close it`、
  `libero_90` の `close the microwave`。
- openpi の `pi05_libero` の学習スイートは、記憶では spatial/object/goal/10（libero_90 は含まない）。
  **公式の説明では未確認**（手順 4-4 で確かめる）。もし正しければ、「開ける」「取り出す」の技能はどちらも学習にないことになり、
  (b) はほぼ 0 という予想が強まる。

### 決まりとの食い違い（報告）
- このサーバーの `/home/ubuntu/slocal1/third_party/Libero-Occ/benchmark_assets/` に LIBERO-Occ の
  BDDL・初期状態がすでに置かれている（本線 StowSafe 用）。依頼の「このサーバーに置かない」と食い違う。
  この探索実験では読み込まない。削除するかは Kevin の判断待ち（触っていない）。
