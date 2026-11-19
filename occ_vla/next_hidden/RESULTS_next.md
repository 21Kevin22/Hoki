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

## 2026-11-19 手順 4 の続き：π0.5 の重みと学習スイート（公式で確認）
- openpi：`third_party/openpi`（commit `215abfb217dbac7d5f1273282331b9b1866c0479`）、`uv sync` で導入。
- 重み：`gs://openpi-assets/checkpoints/pi05_libero`（openpi README の π0.5-LIBERO）。16 ファイル、
  ファイルごとの sha256 は `results/MANIFEST_pi05_libero_weights.json`、まとめの sha256 は
  `0b289bbe95965295cb7aa649b417f7226c0e399cfa7d92b7da3b4aa26535a18a`。
- 学習データ：`pi05_libero` の設定の `repo_id="physical-intelligence/libero"`。その `meta/tasks.jsonl` は 40 タスク
  （spatial/object/goal/10。**libero_90 は含まない**）。`results/pi05_libero_train_tasks.jsonl`。

## 2026-11-19 6-1 タスクの作成と確認（方策なし、描画のみ GPU 2 で EGL）
**タスク**：`bddl/KITCHEN_SCENE6_take_the_yellow_and_white_mug_out_of_the_microwave_and_put_it_on_the_plate.bddl`。
libero_10 の「マグを電子レンジに入れて閉める」と同じ場面に、皿を 1 枚足しただけ（マグが元いた場所）。
LIBERO の配置器は閉じた電子レンジの中に物を置けないので、`scripts/build_init_states.py` がリセット後の状態を加工する
（マグを中に移す → 扉を閉じる → 静置）。

**成功の判定（コードで確認）**：LIBERO のゴール `On(white_yellow_mug_1, plate_1)`。
`check_ontop`：皿の原点 z ≤ マグの原点 z、かつ皿とマグが接触、かつ両原点の xy 距離 < 3 cm。エピソードの `done` はこれ。

**初期状態の確認**：10 個（21 回試行）。残す条件：閉（LIBERO の Close）、中（In）、ゴール偽、マスク画像で 0 画素（両カメラ）、
RGB で 0 画素（256×256、両カメラ）、静置（40 ステップで < 2 mm）、めり込み < 1 mm、扉 < 0.3° かつマグが扉に触れない、
読み込み直し 3 回（1 回は解法器の warm-start を消す）でも全部通る。

| 確認 | 結果 |
|---|---|
| 読み込み直し（同じ env で順に＋warm-start を消して） | 19 / 20（状態 6 が warm-start を消すと 4.1 mm 滑る） |
| マスク画像（セグメンテーション）の画素 | 全状態・両カメラ 0 |
| RGB 画素（後から別に測定） | 手首 全状態 0。正面 6/10 状態で 0、4 状態で 1〜18 画素（実行ごとに変わる） |

### 途中で見つけて直したこと
1. 皿の場所がマグの初期の場所と重なり、リセット時に接触 → 皿をマグの元の場所に、マグの（捨てる）初期領域を手前右に。
2. マグを床に落とす高さを間違え、床と天井に 2 cm めり込み → マグの底からの距離を測って置く。
3. マグが前寄りだと取っ手が扉の内側に触れ、扉が約 2° 勝手に開いて隙間から見えた → 少し奥に置き、「マグが扉に触れない・扉 < 0.3°」を条件に追加。
4. 読み込み直すと 3〜4 mm 滑る状態があった → 原因は乱数の種ではなく、前のエピソードから残る解法器の warm-start。
   warm-start を消した読み込みも条件に追加（それでも 1 状態は検証で落ちる）。
5. **扉のガラス（材質の不透明度 0.25）**：マスク画像では不透明に描かれるが、RGB ではマグが透けて見える。
   腕が動くと手首カメラの RGB に扉越しのマグが写る（`results/expert/state4_frames.png` の 2 段目）。
   初期状態でも正面カメラの RGB に 1〜18 画素写ることがあり、実行ごとに変わる（扉は 0.057° のまま。マグのわずかな滑りと思われる＝推測）。

## 2026-11-19 6-2 手本の動作（正解の位置を使う、方策なし）← 関門
`scripts/scripted_expert.py`、`results/expert/summary.json`、各状態の動画の抜粋 `results/expert/state*_frames.png`。
手順：扉の縦の取っ手をつまむ → ちょうつがいの周りの弧に沿って開ける → 扉の外側から回り込む（高く通る）→
マグの**取っ手**を横からつまむ（胴は約 9 cm で Panda の最大開き 8 cm より太く、上は天井まで 2 cm 弱で上からつかめない）→
引き出す → 皿の上へ運んで置く。

| 状態 | 成功 | ステップ | 扉の角度 | 取っ手に届いた | マグと皿の xy 距離 | 電子レンジとの接触（取っ手以外） |
|---|---|---|---|---|---|---|
| 0 | ✓ | 601 | 103.5° | ✓ | 2.1 cm | 5 |
| 1 | ✗ | 597 | 103.5° | ✓ | 9.2 cm | 3 |
| 2 | ✓ | 598 | 103.5° | ✓ | 1.5 cm | 4 |
| 3 | ✓ | 599 | 103.5° | ✓ | 1.2 cm | 1 |
| 4 | ✗ | 604 | 103.5° | ✓ | 3.2 cm | 0 |
| 5 | ✗ | 603 | 103.5° | ✓ | 3.2 cm | 0 |
| 6 | ✓ | 601 | 103.5° | ✓ | 2.5 cm | 3 |
| 7 | ✓ | 599 | 103.5° | ✓ | 0.7 cm | 4 |
| 8 | ✓ | 598 | 103.5° | ✓ | 1.5 cm | 2 |
| 9 | ✓ | 597 | 103.5° | ✓ | 2.1 cm | 4 |

**7 / 10 成功**。扉を開ける・取っ手に届く・マグを外に出すは 10/10。失敗 3 本はすべて最後の置く段で、
取っ手でぶら下げたマグが皿の縁に傾いて着き、中心から 3.2〜9.2 cm（判定は 3 cm 未満）。手本のつくりの粗さで、場面の幾何の問題ではない。

**扉の開き具合と入口の幅**（手の真っすぐな進入を扉がふさがない x 方向の幅。入口の全幅は 20.7 cm）：

| 扉の角度 | 30° | 45° | 60° | 76° | 97.5° | 118° |
|---|---|---|---|---|---|---|
| 空いている幅 | 1.2 cm | 5.4 cm | 10.7 cm | 17.4 cm | 20.7 cm | 20.7 cm |

- LIBERO の元のタスクの初期状態（50 個）の扉の角度：76〜118°（平均 97.5°）。手本は 103.5° まで開け、その範囲内。
- LIBERO の Open の判定は 74.5° 超。手本の開き方では全状態で Open が真。
- **狭いのは扉より、マグの取っ手に届くときの手の幅**：Panda の手は指の開く方向に約 20 cm、入口は 20.7 cm。
  取っ手が奥すぎる・左右に寄りすぎると、手が側壁に当たって届かない（置き場所の範囲を狭めて直した。x ±0.8 cm、y ±0.6 cm）。
  残る電子レンジとの接触は、取っ手をつまむときに手が側壁に 0〜5 ステップ触れるもの。
- 手本は、皿に置くときに隣の別のマグ（porcelain mug）に約 170 ステップ触れている（手本の動きの粗さ。直していない）。
- 対象が最初に写るステップ（マスク画像）：正面 72〜73、手首 76 または 114（扉を開け始めて数ステップ後）。

## 2026-11-19 6-3 技能と言葉（方策なし）
`results/instruction_table.json`（学習の 40 タスクと、語の重なりで比べたもの。モデルは使っていない）。

| 使う条件 | 候補の指示文 | 学習にそのままある | 一番近い学習のタスク | 語の重なり |
|---|---|---|---|---|
| A | take the yellow and white mug out of the microwave and put it on the plate | ✗ | put the white mug on the left plate and put the yellow and white mug on the right plate | 0.53 |
| A | open the microwave and put the yellow and white mug on the plate | ✗ | （同上） | 0.67 |
| B/C 1 段目 | open the microwave | ✗ | open the middle drawer of the cabinet | 0.29 |
| B/C 2 段目 | put the yellow and white mug on the plate | ✗ | （A と同じ） | 0.80 |
| B/C 2 段目 | pick up the yellow and white mug and place it on the plate | ✗ | pick up the black bowl on the ramekin and place it on the plate | 0.57 |

- 学習にある電子レンジのタスクは「put the yellow and white mug in the microwave and close it」の 1 つだけ（入れて**閉める**）。
- 「開ける」は引き出しの 2 つだけ。「取り出す（out of / take）」は 0。
- 「マグを皿に置く」は近いもの（同じマグ・同じ言い回し）がある。
- つまり、(b) に要る 3 つの技能のうち、学習にあるのは「置く」だけ。事前の予想（(b) は技能が学習にあれば一部成功）の前提のうち、
  「開ける」「取り出す」は学習に**ない**ことが確かめられた。
