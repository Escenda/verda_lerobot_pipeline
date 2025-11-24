# Verda × LeRobot Spot Training Pipeline

このパッケージは、次の一連の処理を自動化するための最小構成です。

1. Verda (DataCrunch) の Spot インスタンスを指定 GPU 構成でデプロイ
2. SSH で接続し、`.env.remote` を `.env` としてアップロードし、学習用スクリプトを転送
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
  .env.example          # ローカル専用 (DataCrunch 認証など)
  .env.remote.example   # リモートに送る .env のサンプル
  .env.remote           # 実際にリモートへ送る設定 (Git 管理外を推奨)
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

## .env / .env.remote の設定

ローカル用（DataCrunch 認証など）:

```bash
cp .env.example .env
```

主に設定するのは `DATACRUNCH_CLIENT_ID` / `DATACRUNCH_CLIENT_SECRET` などローカルで必要な認証です。

リモート用（学習設定）:

```bash
cp .env.remote.example .env.remote
```

主に設定するのは以下です:

- `HUGGINGFACE_HUB_TOKEN`  
  - private dataset / model / checkpoint repo へのアクセスに使うトークン
- `LEROBOT_DATASET_REPO_ID`  
  - 事前にアップロード済みの HF dataset repo_id (`your-user/your-dataset`)
- `LEROBOT_POLICY_TYPE`  
  - `act` など、使いたいポリシーの種類（SmolVLA なら `policy.path` を併用）
- `LEROBOT_POLICY_PATH`  
  - SmolVLA など事前学習済みチェックポイント（例: `lerobot/smolvla_base`）
- `LEROBOT_RENAME_MAP`, `LEROBOT_POLICY_EMPTY_CAMERAS`  
  - データセットのカメラキーをポリシー期待に合わせる場合に使用
- `LEROBOT_BATCH_SIZE`  
  - バッチサイズ指定
- `LEROBOT_JOB_NAME`, `LEROBOT_OUTPUT_DIR`  
  - 出力ディレクトリ (`outputs/train/...`) の指定
- `LEROBOT_CHECKPOINT_REPO_ID` (任意だが推奨)  
  - `checkpoints/last` をアップロードする Hugging Face model repo_id  
  - ここにアップロードされたチェックポイントから `resume` ができます
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `LEROBOT_WANDB_ENABLE`
- `LEROBOT_TORCH_NIGHTLY` (任意)  
  - B200(sm_100) など最新 GPU で nightly PyTorch/cu128 を上書きしたい場合は `1` を設定

---

## 1. 初回学習（train）

例: H100×4 の Spot インスタンスを立てて学習を開始する:

```bash
cd verda_lerobot_pipeline

python local/verda_lerobot_manager.py train \
  --gpu-model H100 \
  --gpus-per-instance 4 \
  --env-file .env.remote \
  --run-id first_run
```

フロー:

1. Verda API 経由で Spot インスタンスを作成 (`contract=SPOT`)  
2. IP 付与後に SSH で接続  
3. `remote/setup_env.sh`, `remote/train_lerobot_entry.py`, `.env.remote`（リモートでは `.env` として保存）を `/root/lerobot_run` に転送  
4. `setup_env.sh train` を実行  
   - venv 作成 → `lerobot[smolvla]` + 依存をインストール（必要に応じて nightly PyTorch/cu128）  
   - `train_lerobot_entry.py --mode train` を起動  
   - `python -m lerobot.scripts.lerobot_train ...` で学習開始  
5. 正常終了すると、`LEROBOT_CHECKPOINT_REPO_ID` が設定されていれば  
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
  --env-file .env.remote \
  --checkpoint-repo-id your-username/lerobot-checkpoint-private-dataset \
  --run-id resume_run_1
```

このときの流れ:

1. 新しい Spot インスタンスをデプロイ
2. SSH で接続し、`.env` とリモートスクリプトを配置
3. `LEROBOT_CHECKPOINT_REPO_ID` を環境変数で渡し、`setup_env.sh resume_hub` 実行
4. `train_lerobot_entry.py --mode resume_hub` が以下を実行:
   1. `snapshot_download(repo_id=LEROBOT_CHECKPOINT_REPO_ID, repo_type="model")` で  
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
- `LEROBOT_STEPS` を大きめに入れておき、複数回 `resume` しても OK

---

## 3. Weights & Biases

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

## 4. 注意点と割り切り

- Spot kill の瞬間までに Hub にアップロードされているのは、
  **「最後に正常終了した run の last checkpoint」だけ** です。
  - kill 前にまだアップロードが走っていない分のステップは失われます
  - それが致命的であれば、
    - `LEROBOT_STEPS` を小さめに区切って、区切りごとに run を終了させる
    - あるいは train スクリプト内に定期的な `upload_folder()` 呼び出しを追加する
  といった、より攻めた構成が必要になります。
- ここでは「構成をできるだけシンプルに保ちつつ、
  *少なくとも最後の正常終了チェックポイントからは再開できる*」ことを優先しています。

---

## 5. ざっくりまとめ

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
