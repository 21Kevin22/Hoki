# Kaggle 実行ガイド（2026-08-17〜）

サーバーが使えない前提で、OpenVLA-OFT + occlusion-recovery 実験
（`thirdparty/openvla-oft/`）を Kaggle Notebook (GPU) 上で回すための手順。

## 使い方

1. Kaggle Notebook を新規作成し、Accelerator を GPU (T4 x2 か P100) に設定。
2. `oft_kaggle_bootstrap.py` の各 `# %%` ブロックを、そのまま1セルずつ
   コピペして上から順に実行する（jupytext 対応エディタなら直接ノートブック
   として開ける）。
3. 初回は Cell 1〜7（環境確認→依存インストール→チェックポイントDL→
   スモークテスト）まで通す。所要時間の大半はチェックポイントDL（約15GB）
   とEGL/依存関係のインストール。
4. Cell 6 で案内している通り、ダウンロードしたチェックポイントは
   **Kaggle Dataset として保存**しておくこと。セッションは終了時に
   `/kaggle/working` が消えるため、保存しないと次回また15GB DLし直しになる。
5. 2回目以降のセッションは Cell 3〜5（clone/install/DL）を飛ばし、保存した
   Dataset を attach してスモークテスト（Cell 7）から再開する。

## GPUクォータについて

Kaggle は週30時間（T4/P100共有）。フルスケールのA5（10タスク×3seed×
50rollout×2条件×2ベンチ）は1週間分のクォータでは到底終わらない前提で、
`--start-episode`（`run_oft_camera_dropout_eval.py` には未実装 — 追加が
必要、下記TODO参照）で分割・再開する運用を想定。

まずは小規模（A1, A4, B2, B3, B1のパイロット）から着手し、感触を見てから
A5のスケジュールを組むのが現実的。

## このセッションで実装したもの（ローカルMacでユニットテスト済み、
## Kaggle上での実機動作は未検証）

- `scripts/oft_occlusion_gate.py` — デバウンス閾値ゲート（k, threshold,
  mode=threshold/always/never）。B1のk掃引はログ再集計のみで可能。
- `scripts/oft_occlusion_gt.py` — パッチ単位の遮蔽真値（occ_gt）。合成
  オクルージョンなので厳密に計算可能（A3の評価基盤）。
- `scripts/oft_step_logger.py` — 依頼されたロギング仕様(step, episode,
  task_id, seed, s_occ, occ_flag, debounce_counter, correction_applied,
  occ_gt, ee_position, action, t_vla_ms, t_predictor_ms, t_total_ms,
  success, steps_to_success)を満たすJSONLライター。
- `scripts/oft_timing.py` — A1向けのレイテンシ計測（predictor forward
  のみをラップしてCUDA同期付きで計測）。
- `scripts/run_oft_camera_dropout_eval.py` に統合:
  - `--log-steps-dir`, `--s-occ-source`, `--debounce-threshold`,
    `--debounce-k`, `--gate-mode`, `--gate-no-latch`, `--measure-latency`,
    `--seed` を追加
  - 新条件 `wrist_partial_vjepa_gated`（デバウンスゲート越しに実際に
    補正のON/OFFを切り替える、B1のライブ版）
  - 新条件 `wrist_partial_prevframe`（B3の最重要コントロール：前フレーム
    コピー、追加パラメータ0）

30個のユニットテスト（`scripts/tests/test_oft_*.py`）は全てローカルMac
（GPU不要）で合格済み。**ただし `run_oft_camera_dropout_eval.py` 本体への
統合部分は、LIBERO/CUDA環境がこのマシンに無いため実機で1回も動かせて
いない** — Kaggleでの最初の実行が事実上の初回テストになる。

## 重要な注意（S_occ について）

現状 `--s-occ-source oracle`（デフォルト）は S_occ = occ_gt（真値そのもの）
としている。これはロギング/ゲート機構そのものの配線確認、およびA1/A2/A4/B1
には使えるが、**A3（検知の適合率・再現率）をこのモードで計算すると
定義上100%になり、実際の検知性能の評価にはならない**。真のS_occ検知器
（`thirdparty/openvla-oft/CLAUDE.md`の"In progress"にある隠れ状態プローブ）
は未完成のため、`--s-occ-source probe` は `NotImplementedError` を返す
実装のまま残してある。A3を本当に評価するには、このプローブの学習を
先に終わらせる必要がある。

## TODO（次にやると良い順）

1. Kaggle上でCell 7のスモークテストを1回通し、ログの中身を目視確認する
   （Cell 8）— ロジックはローカルで検証済みだが、実データでの動作は未確認
2. 動いたら `--num-trials` を増やして A1（レイテンシ）と B3（前フレーム
   コピー vs. 学習済みpredictor）から本実験に着手
3. `--start-episode` 的な再開機能を `run_oft_camera_dropout_eval.py` に
   追加（週次クォータをまたぐ分割実行のため、`collect_failure_probe_data.py`
   に既にある `--start-episode` パターンを移植すればよい）
4. S_occ検知器（失敗予測プローブ）の学習スクリプトを書く → A2/A3が現実になる
