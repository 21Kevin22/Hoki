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
（GPU不要）で合格済み。**2026-08-18、実際にKaggle上でスモークテストが
最後まで通り（4bit量子化込み、`success=True`）、ロギング/デバウンスゲート
パイプライン全体が実機で動作確認済み。** 通るまでに実際に踏んだ問題と
修正は下記の通り（同じ問題を繰り返さないための記録）。

### 実機で踏んだ問題と修正（2026-08-17〜18、時系列）

1. `run_oft_camera_dropout_eval.py`等9スクリプトが元のプロジェクトサーバー
   のパスをハードコードしていた → `__file__`から相対的に導出するよう修正
2. `pip install -e LIBERO_DIR`（PEP 660デフォルト）がLIBEROのネストした
   `libero/libero/`構造でfinder登録に失敗し`ModuleNotFoundError`
   → `--config-settings editable_mode=compat`（レガシーeditable install）
3. `libero_requirements.txt`のインストールがnumpyを2.2.6・mujocoを3.11.0に
   引き上げ、torch==2.2.0（NumPy 1.x C-API依存）・robosuite==1.4.1
   （`mj_fullM()`の引数仕様）の両方を壊す → インストール後に明示的に
   `numpy==1.26.4 mujoco==3.0.0`へ固定し直す
4. Jupyterカーネル自身の`MPLBACKEND`環境変数がsubprocessに継承され、
   matplotlibがIPython外でクラッシュ → `env["MPLBACKEND"]="Agg"`を明示
5. `prismatic.vla`パッケージの`__init__.py`が推論に不要なRLDS学習データ
   読み込み一式（tensorflow/tensorflow_datasets/dlimp）を無条件import、
   protobufバージョン衝突（`tensorflow==2.15.0`の上限 vs
   `tensorflow_metadata`が要求する`runtime_version`はprotobuf>=5.26必須）
   で解決不能 → 該当importをtry/exceptで遅延化。さらに`openvla_utils.py`
   も同じ連鎖を`NormalizationType`経由で引いていたので、軽量な
   `prismatic.vla.constants`から直接importするよう変更
6. 7Bモデルがbf16のままだと16GBのT4でOOM → `--load-in-4bit`/
   `--load-in-8bit`をCLIに追加
7. `bitsandbytes`を無指定でinstallすると最新版がtorchを2.2.0→2.13.0に
   勝手に引き上げてtorchvisionと非互換に → `bitsandbytes==0.43.1`
   （torch 2.2.0時代のバージョン）を`--no-deps`で固定
8. 量子化時、`vjepa_predictor_dino`/`_siglip`（チェックポイントに無い
   新規パラメータ）がuint8型のmetaテンソルとして生成され、
   `reset_parameters()`のCUDA正規分布初期化が失敗
   → `nn.Module.to(dtype=...)`は「既に浮動小数点型のテンソルにしか
   dtypeを適用しない」仕様のため無効。各パラメータの`.data`を直接
   上書きして回避
9. `device_map`が単一デバイスに解決される場合、常に
   `"`.to` is not supported for 4-bit/8-bit bitsandbytes models"`で
   失敗（`None`/`{"":torch.device}`/`{"":0}`/`quantization_config`
   経由、全パターンで再現） → 根本原因は`accelerate`のバージョン
   （pipが最新の1.14.0を入れていたが、`transformers`カスタムフォーク
   は2024年半ば・accelerate 0.3x系を前提）。`accelerate==0.30.1`に
   固定して解決
10. `run()`ヘルパーは`capture_output=True`で全出力をバッファするため、
    実際には正常に数分かかっているだけの実行が「ハングしている」ように
    見えた → 長時間実行するスモークテストは`subprocess.Popen`で
    リアルタイムにstdoutをストリーミングする方式に変更（Cell 7に反映済み）

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

1. ~~Kaggle上でCell 7のスモークテストを1回通し、ログの中身を目視確認する~~
   — **2026-08-18完了**（`baseline`条件で`success=True done_step=376`）。
   残り3条件（`wrist_partial`, `wrist_partial_vjepa_gated`,
   `wrist_partial_prevframe`）とログ中身の目視確認はこれから。
2. `--num-trials` を増やして A1（レイテンシ）と B3（前フレームコピー vs.
   学習済みpredictor）から本実験に着手
3. `--start-episode` 的な再開機能を `run_oft_camera_dropout_eval.py` に
   追加（週次クォータをまたぐ分割実行のため、`collect_failure_probe_data.py`
   に既にある `--start-episode` パターンを移植すればよい）
4. S_occ検知器（失敗予測プローブ）の学習スクリプトを書く → A2/A3が現実になる
