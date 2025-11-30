# Verda × LeRobot Spot Training Pipeline

このパッケージは、次の一連の処理を自動化するための最小構成です。

1. Verda (DataCrunch) の Spot インスタンスを指定 GPU 構成でデプロイ
2. SSH で接続し、`.env`（シークレット）と `train.remote.yaml`（学習設定）をアップロードし、学習用スクリプトを転送
3. リモートで LeRobot の環境構築 (venv + `pip install lerobot`)
4. Hugging Face Hub 上の **プライベートデータセット** を使って `lerobot` の学習を開始
5. `wandb` に学習ログを送信
6. 学習完了後、最新のチェックポイント (`checkpoints/last`) を Hugging Face Hub の **model repo** にアップロード
7. Spot インスタンスが kill された場合、別インスタンスから **Hub 上のチェックポイント repo_id を指定して学習を再開**
8. 正常終了時には、リモートの `output_dir` 以下を丸ごとローカルにダウンロード
9. 最後に Verda インスタンスを削除して課金を停止

---

## ファイル構成

```text
verda_lerobot_pipeline/
  .env.example              # ローカル/リモートで使うシークレットのサンプル
  .env                      # 実際のシークレット (Git 管理外)
  configs/
    train.remote.example.yaml # リモート学習用の汎用サンプル (YAML)
  local/
    verda_lerobot_manager.py  # ローカルで実行するオーケストレータ
  remote/
    setup_env.sh  # リモート側で venv 構築 & 依存関係インストール
    train_lerobot_entry.py    # LeRobot 学習 & Hub チェックポイント連携
```

---

## 前提条件

ローカル環境で必要なもの:

```bash
pip install datacrunch paramiko huggingface_hub python-dotenv
```

認証は `.env`（ローカル用）に記載し、Git にはコミットしないでください。

---

## .env と train.remote.yaml の設定

シークレット（.env）:

```bash
cp .env.example .env
```

- `DATACRUNCH_CLIENT_ID`, `DATACRUNCH_CLIENT_SECRET`（Verda API 認証）
- `VERDA_SSH_KEY_NAME`, `VERDA_SSH_PRIVATE_KEY`（ローカル→リモートの SSH で使用）
- `HUGGINGFACE_HUB_TOKEN`（private dataset/model へのアクセス）
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `LEROBOT_WANDB_ENABLE`（wandb 用）

トレーニング設定（YAML）:

```bash
cp configs/train.remote.example.yaml configs/train.remote.yaml
```

用途別に実値を入れたサンプルも用意しました。必要に応じて `repo_id` だけ自分の名前空間に変えてください。

- ACT: `configs/train.remote.act.sample.yaml`
- SmolVLA: `configs/train.remote.smolvla.sample.yaml`
- Pi0: `configs/train.remote.pi0.sample.yaml`

主に設定するキー:

- `dataset.repo_id`  
  - 事前にアップロード済みの HF dataset repo_id (`your-user/your-dataset`)
- `policy.*`  
  - `type` / `path` / `pretrained_path` / `rename_map` / `empty_cameras` / `repo_id` など
- `job.name`, `job.output_dir`  
  - 出力ディレクトリ (`outputs/train/...`) の指定
- `training.steps`, `training.batch_size`, `training.save_freq`, `training.save_checkpoint`
- `checkpoint.repo_id`（任意だが推奨）  
  - `checkpoints/last` をアップロードする Hugging Face model repo_id  
  - `upload_every_save` を true にすると `save_freq` ごとに `checkpoints/last` を Hub に同期（Spot kill 対策）  
    `upload_poll_seconds` で監視ポーリング間隔を秒指定（デフォルト60）
- `wandb.enable`（API key は .env 側で設定）
- `accelerate.*`（`use`, `num_processes`, `num_machines`, `mixed_precision`, `config_file`）
- `torch.nightly`（B200 等で nightly cu128 を入れたいとき true）

デフォルトの YAML 名は `configs/train.remote.yaml`。別名を使う場合は CLI の `--train-config` か、`.env` に `LEROBOT_TRAIN_CONFIG` を設定してください。

---

## 1. 初回学習（train）

例: H100×4 の Spot インスタンスを立てて学習を開始する:

```bash
cd verda_lerobot_pipeline

python local/verda_lerobot_manager.py train \
  --gpu-model H100 \
  --gpus-per-instance 4 \
  --env-file .env \
  --train-config configs/train.remote.yaml \
  --run-id first_run
```

フロー:

1. Verda API 経由で Spot インスタンスを作成 (`contract=SPOT`)  
2. IP 付与後に SSH で接続  
3. `remote/setup_env.sh`, `remote/train_lerobot_entry.py`, `.env`（シークレット）と `train.remote.yaml` を `/root/lerobot_run` に転送  
4. `setup_env.sh train` を実行  
   - venv 作成 → `lerobot[smolvla]` + 依存をインストール（必要に応じて nightly PyTorch/cu128）  
   - `train_lerobot_entry.py --mode train --config train.remote.yaml` を起動  
   - `python -m lerobot.scripts.lerobot_train ...` で学習開始  
5. 正常終了すると、`checkpoint.repo_id`（もしくは `LEROBOT_CHECKPOINT_REPO_ID`）が設定されていれば  
   `OUTPUT_DIR/checkpoints/last` が Hugging Face Hub にアップロードされます  
6. ローカル側では `runs/{run_id}/outputs/` 以下に `OUTPUT_DIR` が再帰コピーされます  
7. 最後にインスタンスを `delete` して課金を止めます

---

## 2. Spot kill や中断後に再開（resume）

事前に、少なくとも一度は `checkpoints/last` を Hub にアップロードしておく必要があります。  
（初回 run が正常終了した時点で自動アップロードされます）

再開コマンド例:

```bash
python local/verda_lerobot_manager.py resume \
  --gpu-model H100 \
  --gpus-per-instance 4 \
  --env-file .env \
  --train-config configs/train.remote.yaml \
  --checkpoint-repo-id your-username/lerobot-checkpoint-private-dataset \
  --run-id resume_run_1
```

このときの流れ:

1. 新しい Spot インスタンスをデプロイ
2. SSH で接続し、`.env` とリモートスクリプトを配置
3. `checkpoint.repo_id`（または CLI の `--checkpoint-repo-id` で渡した値）が設定された状態で `setup_env.sh resume_hub` を実行
4. `train_lerobot_entry.py --mode resume_hub` が以下を実行:
   1. `snapshot_download(repo_id=checkpoint.repo_id, repo_type="model")` で  
      Hub 上のチェックポイントをローカルに展開
   2. その中の `pretrained_model/train_config.json` から `output_dir` を読み出し、
      `output_dir/checkpoints/last/{pretrained_model,training_state}` を再構築
   3. `python -m lerobot.scripts.train --config_path=.../pretrained_model --resume=true` で学習再開
   4. 正常終了したら新しい `checkpoints/last` を再び同じ repo_id に `upload_folder()` でアップロード
5. ローカル側で再び `runs/{run_id}/outputs/` に `output_dir` をコピー
6. インスタンスを削除

これにより、

- Spot インスタンスが kill されても  
  「最後に Hub にアップロードされたチェックポイント」から再開可能
- `training.steps` を大きめに入れておき、複数回 `resume` しても OK

---

## 3. ローカルストレージからの再開（resume_local）

Spot kill されたインスタンスでも OS ボリュームは残ります。`resume_local` ではそのボリュームから新しいインスタンスを起動し、ローカルに残った `checkpoints/last` を探索して再開します。

再開コマンド例（ボリュームは対話選択; ID が分かれば `--os-volume-id` を指定）:

```bash
python local/verda_lerobot_manager.py resume_local \
  --gpu-model H100 \
  --gpus-per-instance 4 \
  --env-file .env \
  --train-config configs/train.remote.yaml \
  --run-id resume_local_1
```

流れ:

1. `--os-volume-id` があればそれを、なければ「未接続の OS ボリューム」を一覧表示して選択  
   - ボリュームの場所（FIN-0x/ICE-01）に合わせて同じロケーションで Spot インスタンスを作成  
   - `--image` は無視され、選択した OS ボリュームでブート
2. `.env` と `train.remote.yaml`、リモートスクリプトを `/root/lerobot_run` に再配置
3. `setup_env.sh resume_local` → `train_lerobot_entry.py --mode resume_local` を実行  
   - まず `job.output_dir`（YAML）にある `checkpoints/last` を探し、なければ `outputs/train/**/checkpoints/last` を横断検索  
   - `training_state/training_step.json` のステップが最も進んでいるものを選択し、`--resume=true` で再開
4. `checkpoint.repo_id`（または CLI `--checkpoint-repo-id`）が設定されていれば、再開後の `checkpoints/last` を Hub へ再アップロード
5. ローカルへ `output_dir` をダウンロードし、インスタンスを削除

---

## 4. Weights & Biases

`.env` に以下を設定し、`LEROBOT_WANDB_ENABLE=true` としておけば、
LeRobot 側が自動で wandb にログを送ります。

```env
WANDB_API_KEY=...
WANDB_PROJECT=lerobot-training
WANDB_ENTITY=your-entity
LEROBOT_WANDB_ENABLE=true
```

`train_lerobot_entry.py` からは `--wandb.enable=true/false` を渡すだけなので、  
wandb の詳細な挙動（project, entity など）は wandb 側の環境変数設定に従います。

---

## 5. マルチGPU（accelerate）

- デフォルト動作: GPU を2枚以上検出した場合、自動で `accelerate launch --multi_gpu --num_processes=<検出GPU数> --module lerobot.scripts.lerobot_train ...` で実行します。
- 強制 ON/OFF: `accelerate.use: true/false` を YAML に記載（`LEROBOT_USE_ACCELERATE` でも上書き可、`false` なら単一プロセス実行）。
- 使用するプロセス数を固定したい場合: `accelerate.num_processes: 4`（もしくは環境変数 `LEROBOT_ACCELERATE_NUM_PROCESSES`）。
- mixed precision の指定: `accelerate.mixed_precision: bf16`（`fp16`/`bf16`/`no`。環境変数 `LEROBOT_ACCELERATE_MIXED_PRECISION` でも可）。
- 既存の accelerate 設定ファイルを使いたい場合: `accelerate.config_file: /root/accelerate_config.yaml`（環境変数 `LEROBOT_ACCELERATE_CONFIG_FILE` でも指定可）。
- 学習率やステップ数は LeRobot 側で自動スケールされません。必要に応じて [公式の Multi-GPU ガイド](https://huggingface.co/docs/lerobot/multi_gpu_training)の通り、手動でスケーリングしてください（effective batch size = batch_size × GPU 数）。

---

## 6. Pi0 でのトレーニング

- 依存インストール: `lerobot[smolvla,pi]` をリモートで入れるよう変更済み（`setup_env.sh` で自動）。
- `configs/train.remote.example.yaml` に SmolVLA/ACT/Pi0 用のプリセットブロックを用意したので、必要なものだけ反映して `configs/train.remote.yaml` を作ってください。
- Pi0 の推奨設定（YAML の `policy` セクションで指定）:
  - `type: pi0`
  - `pretrained_path: lerobot/pi0_base`（または `lerobot/pi0_libero`）
  - `compile_model: true`
  - `gradient_checkpointing: true`
  - `dtype: bfloat16`
- 上記が未指定でも、`policy.type=pi0` の場合はデフォルトで `pretrained_path=lerobot/pi0_base`, `compile_model=true`, `gradient_checkpointing=true`, `dtype=bfloat16` を付与して起動します。
- 例: `configs/train.remote.yaml` で `policy.type: pi0` を指定し、通常通り `verda_lerobot_manager.py train ...` を実行すると Pi0 で学習が走ります。

---

## 7. 注意点と割り切り

- Spot kill の瞬間までに Hub にアップロードされているのは、
  **「最後に正常終了した run の last checkpoint」だけ** です。
  - kill 前にまだアップロードが走っていない分のステップは失われます
  - それが致命的であれば、
    - `checkpoint.repo_id` を設定し、`checkpoint.upload_every_save: true` にすると `save_freq` 間隔で Hub に `checkpoints/last` を同期（kill 後も直近ステップまで復元可能）
    - 併せて `training.save_freq` を短くし、保存間隔そのものを細かくする
    - さらに厳密に区切りたい場合は `training.steps` を小さめに設定し、run を細かく終端させる
  といった、より攻めた構成が必要になります。
- ここでは「構成をできるだけシンプルに保ちつつ、
  *少なくとも最後の正常終了チェックポイントからは再開できる*」ことを優先しています。

---

## 8. ざっくりまとめ

- ローカル: `verda_lerobot_manager.py` を叩くだけで
  - Verda Spot デプロイ → SSH → 学習 → ログ取得 → インスタンス削除 まで自動
- リモート: `setup_env.sh` + `train_lerobot_entry.py` が
  - venv 構築、LeRobot/Hub/wandb 設定、学習再開ロジック、Hub への checkpoint upload を担当
- Hugging Face:
  - dataset repo: 入力データ
  - checkpoint model repo: optimizer 状態ごと last checkpoint を保存
  - policy repo (任意): 推論用にきれいなポリシーだけを公開

あとは `.env` と CLI の引数を自分の環境に合わせて調整すれば、
かなりストレス少なく「Spot 前提のロボット学習パイプライン」が回せるはずです。
