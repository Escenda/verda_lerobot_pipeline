import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from huggingface_hub import HfApi, snapshot_download, upload_folder


def log(msg: str) -> None:
  print(f"[LE-REMOTE] {msg}", flush=True)


def env_bool(name: str, default: bool = False) -> bool:
  v = os.getenv(name)
  if v is None:
    return default
  v = v.strip().lower()
  return v in ("1", "true", "yes", "y", "on")


def env_int(name: str) -> Optional[int]:
  v = os.getenv(name)
  if v is None:
    return None
  try:
    return int(v)
  except ValueError:
    log(f"[WARN] {name} は整数で指定してください: {v!r}")
    return None


def coerce_bool(val: Any) -> Optional[bool]:
  if isinstance(val, bool):
    return val
  if isinstance(val, (int, float)):
    return bool(val)
  if isinstance(val, str):
    v = val.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
      return True
    if v in ("0", "false", "no", "n", "off"):
      return False
  return None


def coerce_int(val: Any, *, name: Optional[str] = None) -> Optional[int]:
  if val is None:
    return None
  if isinstance(val, int):
    return val
  if isinstance(val, str):
    v = val.strip()
    if not v:
      return None
    try:
      return int(v)
    except ValueError:
      if name:
        log(f"[WARN] {name} は整数で指定してください: {val!r}")
      return None
  return None


def load_training_config(path: str) -> Dict[str, Any]:
  cfg_path = Path(path)
  if not cfg_path.is_file():
    raise RuntimeError(f"Training config YAML not found: {cfg_path}")

  with cfg_path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

  if not isinstance(data, dict):
    raise RuntimeError("Training config YAML のトップレベルは mapping である必要があります。")

  return data


def cfg_get(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
  cur: Any = cfg
  for key in dotted_key.split("."):
    if not isinstance(cur, dict) or key not in cur:
      return default
    cur = cur[key]
  return cur


def ensure_hf_tokens() -> None:
  token = os.getenv("HUGGINGFACE_HUB_TOKEN")
  if not token:
    raise RuntimeError("HUGGINGFACE_HUB_TOKEN must be set in the environment for private Hub access.")

  # Map to the standard names used by Hugging Face libraries
  if not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = token
  if not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def run_cmd(cmd: list[str]) -> None:
  log(f"Running command: {' '.join(cmd)}")
  proc = subprocess.Popen(cmd)
  proc.communicate()
  if proc.returncode != 0:
    raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def run_cmd_with_optional_upload(
  cmd: list[str],
  *,
  output_dir: str,
  checkpoint_repo_id: Optional[str],
  enable_upload: bool,
  poll_seconds: int,
) -> None:
  """学習プロセスを起動し、必要ならバックグラウンドで checkpoints/last を Hub へ同期する。"""
  if not (enable_upload and checkpoint_repo_id):
    run_cmd(cmd)
    return

  stop_event = threading.Event()
  watcher = threading.Thread(
    target=watch_and_upload_last_checkpoint,
    args=(output_dir, checkpoint_repo_id, poll_seconds, stop_event),
    daemon=True,
  )

  log(
    f"Start checkpoint watcher: upload checkpoints/last every {poll_seconds}s "
    f"to repo {checkpoint_repo_id}"
  )
  watcher.start()
  try:
    run_cmd(cmd)
  finally:
    stop_event.set()
    watcher.join(timeout=30)
    log("Checkpoint watcher stopped.")


def compute_output_dir(dataset_repo_id: str, policy_type: str) -> str:
  safe = dataset_repo_id.replace("/", "__").replace(".", "_")
  return f"outputs/train/{policy_type}_{safe}"


def normalize_rename_map(val: Any) -> Optional[str]:
  if val is None:
    return None
  if isinstance(val, str):
    return val
  if isinstance(val, dict):
    try:
      return json.dumps(val)
    except Exception as e:  # noqa: BLE001
      log(f"[WARN] rename_map を JSON 化できませんでした: {e}")
      return None
  return str(val)


def write_output_dir_file(output_dir: str) -> None:
  path = Path("output_dir.txt")
  path.write_text(str(Path(output_dir).resolve()), encoding="utf-8")
  log(f"Wrote output_dir to {path}: {path.read_text(encoding='utf-8')}")


def upload_last_checkpoint(output_dir: str, checkpoint_repo_id: str) -> None:
  if not checkpoint_repo_id:
    return

  last_dir = Path(output_dir) / "checkpoints" / "last"
  if not last_dir.is_dir():
    log(f"[WARN] No last checkpoint directory at {last_dir}, nothing to upload.")
    return

  # We expect the "last" directory to contain both pretrained_model/ and training_state/
  log(f"Uploading last checkpoint from {last_dir} to Hub repo {checkpoint_repo_id} (type=model)")
  api = HfApi()
  api.create_repo(repo_id=checkpoint_repo_id, repo_type="model", exist_ok=True)

  upload_folder(
    folder_path=str(last_dir),
    repo_id=checkpoint_repo_id,
    repo_type="model",
    commit_message="Update LeRobot checkpoint (last)",
  )
  log("Upload complete.")


def read_last_step(output_dir: str) -> Optional[int]:
  step_file = Path(output_dir) / "checkpoints" / "last" / "training_state" / "training_step.json"
  if not step_file.is_file():
    return None
  try:
    with step_file.open("r", encoding="utf-8") as f:
      data = json.load(f)
    return int(data.get("step"))
  except Exception as e:  # noqa: BLE001
    log(f"[WARN] training_step.json の読み取りに失敗しました: {e}")
    return None


def watch_and_upload_last_checkpoint(
  output_dir: str, checkpoint_repo_id: str, poll_seconds: int, stop_event: threading.Event
) -> None:
  """定期ポーリングで checkpoints/last の更新を検知し、都度 Hub にアップロードする。"""
  ensure_hf_tokens()
  last_uploaded_step: Optional[int] = None

  while not stop_event.is_set():
    step = read_last_step(output_dir)
    if step is not None and step != last_uploaded_step:
      try:
        log(f"Detected new checkpoint at step {step}, uploading to {checkpoint_repo_id}")
        upload_last_checkpoint(output_dir, checkpoint_repo_id)
        last_uploaded_step = step
      except Exception as e:  # noqa: BLE001
        log(f"[WARN] チェックポイントのアップロードに失敗しましたが継続します: {e}")
    stop_event.wait(poll_seconds)


def prepare_output_dir_from_hub_checkpoint(checkpoint_repo_id: str) -> Tuple[str, str]:
  """Download a checkpoint repo from the Hub and reconstruct a local output_dir.

  The repo is expected to contain a copy of a single 'last' checkpoint directory
  (with subdirs pretrained_model/ and training_state/). We then:

  - read train_config.json from pretrained_model/
  - get output_dir from that config
  - create output_dir/checkpoints/last/{pretrained_model,training_state}
    and copy files there
  - return (output_dir, config_path_dir_to_pretrained_model)
  """
  ensure_hf_tokens()

  cache_dir = Path.cwd() / "hf_ckpt_cache"
  cache_dir.mkdir(parents=True, exist_ok=True)

  log(f"Downloading checkpoint repo {checkpoint_repo_id} into {cache_dir}")
  local_root = Path(
    snapshot_download(
      repo_id=checkpoint_repo_id,
      repo_type="model",
      local_dir=str(cache_dir),
      local_dir_use_symlinks=False,
    )
  )

  pretrained_src = local_root / "pretrained_model"
  train_config = pretrained_src / "train_config.json"
  if not train_config.is_file():
    raise RuntimeError(
      f"train_config.json not found at {train_config}. "
      f"Expected a repo layout matching a single 'last' checkpoint."
    )

  with train_config.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

  output_dir = cfg.get("output_dir", "outputs/train/resumed_run")
  output_dir_path = Path(output_dir)

  last_dir = output_dir_path / "checkpoints" / "last"
  pm_target = last_dir / "pretrained_model"
  ts_target = last_dir / "training_state"

  log(f"Reconstructing output_dir={output_dir_path} from Hub snapshot.")
  last_dir.mkdir(parents=True, exist_ok=True)

  # Copy pretrained_model
  if pm_target.exists():
    shutil.rmtree(pm_target)
  shutil.copytree(pretrained_src, pm_target)

  # Copy training_state if present
  ts_src = local_root / "training_state"
  if ts_src.is_dir():
    if ts_target.exists():
      shutil.rmtree(ts_target)
    shutil.copytree(ts_src, ts_target)
  else:
    log("[WARN] training_state directory not found in checkpoint repo; "
        "resume will only restore model weights, not optimizer state.")

  config_path_dir = str(pm_target)
  return str(output_dir_path), config_path_dir


def detect_num_gpus() -> int:
  try:
    import torch

    count = torch.cuda.device_count()
  except Exception as e:  # noqa: BLE001
    log(f"[WARN] GPU数の自動検出に失敗しました: {e}")
    return 0

  if count > 0:
    log(f"CUDA デバイスを {count} 基検出しました。")
  else:
    log("CUDA デバイスが見つかりませんでした。")
  return count


def should_use_accelerate(gpu_count: int, force: Optional[bool]) -> bool:
  env_force = os.getenv("LEROBOT_USE_ACCELERATE")
  if force is not None:
    return force
  if env_force is not None:
    return env_bool("LEROBOT_USE_ACCELERATE")
  return gpu_count > 1


def build_accelerate_launch_prefix(gpu_count: int, acc_cfg: Dict[str, Any]) -> list[str]:
  if not should_use_accelerate(gpu_count, coerce_bool(acc_cfg.get("use"))):
    return []

  if shutil.which("accelerate") is None:
    log("[WARN] accelerate コマンドが見つかりません。単一プロセス実行にフォールバックします。")
    return []

  acc_cmd = ["accelerate", "launch"]

  config_file = acc_cfg.get("config_file") or os.getenv("LEROBOT_ACCELERATE_CONFIG_FILE")
  mixed_precision = acc_cfg.get("mixed_precision") or os.getenv("LEROBOT_ACCELERATE_MIXED_PRECISION")
  num_machines = coerce_int(acc_cfg.get("num_machines"))
  if num_machines is None:
    num_machines = coerce_int(os.getenv("LEROBOT_ACCELERATE_NUM_MACHINES"))

  num_processes = coerce_int(acc_cfg.get("num_processes"))
  if num_processes is None:
    num_processes = coerce_int(os.getenv("LEROBOT_ACCELERATE_NUM_PROCESSES"), name="LEROBOT_ACCELERATE_NUM_PROCESSES")
  if num_processes is None:
    if gpu_count > 0:
      num_processes = gpu_count
    else:
      # accelerate を強制したが GPU 数が検出できない場合でも最低1プロセスは確保する
      num_processes = 1

  if config_file:
    acc_cmd.extend(["--config_file", config_file])
  else:
    if num_machines:
      acc_cmd.append(f"--num_machines={num_machines}")
    elif num_processes:
      acc_cmd.append("--num_machines=1")

    if num_processes:
      acc_cmd.append(f"--num_processes={num_processes}")
      if num_processes > 1:
        acc_cmd.append("--multi_gpu")

    if mixed_precision:
      acc_cmd.append(f"--mixed_precision={mixed_precision}")

  acc_cmd.extend(["--module", "lerobot.scripts.lerobot_train"])
  log(f"accelerate launch で実行します: {' '.join(acc_cmd)}")
  return acc_cmd


def build_train_cmd(gpu_count: int, train_args: list[str], acc_cfg: Dict[str, Any]) -> list[str]:
  acc_prefix = build_accelerate_launch_prefix(gpu_count, acc_cfg)
  if acc_prefix:
    return acc_prefix + train_args
  return [sys.executable, "-m", "lerobot.scripts.lerobot_train"] + train_args


def lerobot_train(mode: str, config_path: str) -> None:
  ensure_hf_tokens()

  cfg = load_training_config(config_path)

  dataset_repo_id = cfg_get(cfg, "dataset.repo_id") or os.environ.get("LEROBOT_DATASET_REPO_ID")
  if not dataset_repo_id:
    raise RuntimeError("LEROBOT_DATASET_REPO_ID (or dataset.repo_id in YAML) is required.")

  policy_type = (
    cfg_get(cfg, "policy.type")
    or os.environ.get("LEROBOT_POLICY_TYPE")
    or "act"
  )
  policy_path = cfg_get(cfg, "policy.path") or os.environ.get("LEROBOT_POLICY_PATH")
  policy_pretrained_path = (
    cfg_get(cfg, "policy.pretrained_path") or os.environ.get("LEROBOT_POLICY_PRETRAINED_PATH")
  )
  policy_dtype = cfg_get(cfg, "policy.dtype") or os.environ.get("LEROBOT_POLICY_DTYPE")

  policy_compile_model = coerce_bool(cfg_get(cfg, "policy.compile_model"))
  if policy_compile_model is None and os.getenv("LEROBOT_POLICY_COMPILE_MODEL") is not None:
    policy_compile_model = env_bool("LEROBOT_POLICY_COMPILE_MODEL")

  policy_gradient_checkpointing = coerce_bool(cfg_get(cfg, "policy.gradient_checkpointing"))
  if (
    policy_gradient_checkpointing is None
    and os.getenv("LEROBOT_POLICY_GRADIENT_CHECKPOINTING") is not None
  ):
    policy_gradient_checkpointing = env_bool("LEROBOT_POLICY_GRADIENT_CHECKPOINTING")

  device = cfg_get(cfg, "runtime.device") or os.environ.get("LEROBOT_DEVICE", "cuda")
  wandb_enable_cfg = coerce_bool(cfg_get(cfg, "wandb.enable"))
  if wandb_enable_cfg is None:
    wandb_enable = env_bool("LEROBOT_WANDB_ENABLE", default=bool(os.getenv("WANDB_API_KEY")))
  else:
    wandb_enable = wandb_enable_cfg

  steps_val = cfg_get(cfg, "training.steps")
  steps = str(steps_val) if steps_val is not None else os.environ.get("LEROBOT_STEPS")

  batch_size_val = cfg_get(cfg, "training.batch_size")
  batch_size = (
    str(batch_size_val) if batch_size_val is not None else os.environ.get("LEROBOT_BATCH_SIZE")
  )

  rename_map_cfg = normalize_rename_map(cfg_get(cfg, "policy.rename_map"))
  rename_map = rename_map_cfg if rename_map_cfg is not None else os.environ.get("LEROBOT_RENAME_MAP")
  policy_empty_cameras_val = cfg_get(cfg, "policy.empty_cameras")
  if policy_empty_cameras_val is not None:
    policy_empty_cameras = str(policy_empty_cameras_val)
  else:
    policy_empty_cameras = os.environ.get("LEROBOT_POLICY_EMPTY_CAMERAS")

  save_freq_cfg = coerce_int(cfg_get(cfg, "training.save_freq"), name="training.save_freq")
  save_freq = save_freq_cfg if save_freq_cfg is not None else env_int("LEROBOT_SAVE_FREQ")

  save_checkpoint_cfg = coerce_bool(cfg_get(cfg, "training.save_checkpoint"))
  save_checkpoint: Optional[bool] = save_checkpoint_cfg
  if save_checkpoint is None and os.getenv("LEROBOT_SAVE_CHECKPOINT") is not None:
    save_checkpoint = env_bool("LEROBOT_SAVE_CHECKPOINT", default=True)

  # policy upload behaviour
  policy_repo_id = cfg_get(cfg, "policy.repo_id") or os.environ.get("LEROBOT_POLICY_REPO_ID")

  # optional checkpoint repo for persistence
  checkpoint_repo_id = cfg_get(cfg, "checkpoint.repo_id") or os.environ.get(
    "LEROBOT_CHECKPOINT_REPO_ID"
  )
  checkpoint_upload_every_save_cfg = coerce_bool(cfg_get(cfg, "checkpoint.upload_every_save"))
  if checkpoint_upload_every_save_cfg is None:
    checkpoint_upload_every_save = env_bool("LEROBOT_CHECKPOINT_UPLOAD_EVERY_SAVE", default=False)
  else:
    checkpoint_upload_every_save = checkpoint_upload_every_save_cfg

  checkpoint_upload_poll_seconds_cfg = coerce_int(
    cfg_get(cfg, "checkpoint.upload_poll_seconds"), name="checkpoint.upload_poll_seconds"
  )
  checkpoint_upload_poll_seconds = (
    checkpoint_upload_poll_seconds_cfg
    if checkpoint_upload_poll_seconds_cfg is not None
    else env_int("LEROBOT_CHECKPOINT_UPLOAD_POLL_SECONDS") or 60
  )

  # Decide output_dir and job_name
  output_dir = cfg_get(cfg, "job.output_dir") or os.environ.get("LEROBOT_OUTPUT_DIR")
  if not output_dir:
    output_dir = compute_output_dir(dataset_repo_id, policy_type)

  job_name = cfg_get(cfg, "job.name") or os.environ.get("LEROBOT_JOB_NAME")
  if not job_name:
    job_name = f"{policy_type}_{dataset_repo_id.split('/')[-1]}"

  accelerate_cfg: Dict[str, Any] = {
    "use": cfg_get(cfg, "accelerate.use"),
    "num_processes": coerce_int(cfg_get(cfg, "accelerate.num_processes"), name="accelerate.num_processes"),
    "num_machines": coerce_int(cfg_get(cfg, "accelerate.num_machines"), name="accelerate.num_machines"),
    "mixed_precision": cfg_get(cfg, "accelerate.mixed_precision"),
    "config_file": cfg_get(cfg, "accelerate.config_file"),
  }

  # Pi0 推奨デフォルト（環境変数で上書き可）
  if policy_type.lower() == "pi0":
    if not policy_pretrained_path:
      policy_pretrained_path = "lerobot/pi0_base"
    if policy_compile_model is None:
      policy_compile_model = True
    if policy_gradient_checkpointing is None:
      policy_gradient_checkpointing = True
    if not policy_dtype:
      policy_dtype = "bfloat16"

  available_gpus = detect_num_gpus()

  # Persist output_dir for the orchestrator
  write_output_dir_file(output_dir)

  def apply_train_overrides(args: list[str]) -> None:
    if steps:
      args.append(f"--steps={steps}")
    if save_freq is not None:
      args.append(f"--save_freq={save_freq}")
    if save_checkpoint is not None:
      args.append(f"--save_checkpoint={'true' if save_checkpoint else 'false'}")

  if mode == "resume_hub":
    # Resume from a checkpoint stored on the Hub
    checkpoint_repo_id_resume = checkpoint_repo_id
    if not checkpoint_repo_id_resume:
      raise RuntimeError(
        "checkpoint.repo_id (or LEROBOT_CHECKPOINT_REPO_ID) must be set for resume_hub mode."
      )

    log(f"Resuming from Hub checkpoint repo: {checkpoint_repo_id_resume}")
    restored_output_dir, config_path_dir = prepare_output_dir_from_hub_checkpoint(
      checkpoint_repo_id_resume
    )
    # Overwrite output_dir so we are consistent
    output_dir = restored_output_dir
    write_output_dir_file(output_dir)

    train_args = [
      f"--config_path={config_path_dir}",
      "--resume=true",
    ]
    if rename_map:
      train_args.append(f"--rename_map={rename_map}")
    if policy_empty_cameras:
      train_args.append(f"--policy.empty_cameras={policy_empty_cameras}")
    apply_train_overrides(train_args)

    cmd = build_train_cmd(available_gpus, train_args, accelerate_cfg)
    run_cmd_with_optional_upload(
      cmd,
      output_dir=output_dir,
      checkpoint_repo_id=checkpoint_repo_id_resume,
      enable_upload=checkpoint_upload_every_save,
      poll_seconds=checkpoint_upload_poll_seconds,
    )

    # After a successful resume run, push the new last checkpoint
    upload_last_checkpoint(output_dir, checkpoint_repo_id_resume)
    return

  # Default: "train" mode (fresh or local resume)
  last_cfg = Path(output_dir) / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
  if last_cfg.is_file():
    log(f"Existing local checkpoint found at {last_cfg}, resuming locally.")
    config_path_dir = str(last_cfg.parent)
    train_args = [
      f"--config_path={config_path_dir}",
      "--resume=true",
    ]
    apply_train_overrides(train_args)
  else:
    log("No local checkpoint found, starting a new training run.")
    train_args = [
      f"--dataset.repo_id={dataset_repo_id}",
      f"--output_dir={output_dir}",
      f"--job_name={job_name}",
      f"--policy.device={device}",
      f"--wandb.enable={'true' if wandb_enable else 'false'}",
    ]
    if batch_size:
      train_args.append(f"--batch_size={batch_size}")
    if rename_map:
      train_args.append(f"--rename_map={rename_map}")
    if policy_empty_cameras:
      train_args.append(f"--policy.empty_cameras={policy_empty_cameras}")
    if policy_path:
      train_args.append(f"--policy.path={policy_path}")
    else:
      train_args.append(f"--policy.type={policy_type}")
    if policy_pretrained_path:
      train_args.append(f"--policy.pretrained_path={policy_pretrained_path}")
    if policy_compile_model is not None:
      train_args.append(f"--policy.compile_model={'true' if policy_compile_model else 'false'}")
    if policy_gradient_checkpointing is not None:
      train_args.append(
        f"--policy.gradient_checkpointing={'true' if policy_gradient_checkpointing else 'false'}"
      )
    if policy_dtype:
      train_args.append(f"--policy.dtype={policy_dtype}")
    if policy_repo_id:
      train_args.append(f"--policy.repo_id={policy_repo_id}")
    else:
      # Avoid ValueError: 'policy.repo_id' argument missing when push_to_hub is true by default
      train_args.append("--policy.push_to_hub=false")

    apply_train_overrides(train_args)

  cmd = build_train_cmd(available_gpus, train_args, accelerate_cfg)
  run_cmd_with_optional_upload(
    cmd,
    output_dir=output_dir,
    checkpoint_repo_id=checkpoint_repo_id,
    enable_upload=checkpoint_upload_every_save,
    poll_seconds=checkpoint_upload_poll_seconds,
  )

  # After a successful training run, optionally upload the last checkpoint
  if checkpoint_repo_id:
    upload_last_checkpoint(output_dir, checkpoint_repo_id)


def main() -> None:
  parser = argparse.ArgumentParser(description="LeRobot training entry on Verda instance")
  parser.add_argument(
    "--mode",
    choices=["train", "resume_hub"],
    default="train",
    help="train: new run or local resume; resume_hub: resume from checkpoint repo on Hugging Face",
  )
  parser.add_argument(
    "--config",
    default=os.getenv("LEROBOT_TRAIN_CONFIG", "train.remote.yaml"),
    help="Training config YAML file (default: train.remote.yaml or $LEROBOT_TRAIN_CONFIG)",
  )
  args = parser.parse_args()
  lerobot_train(args.mode, args.config)


if __name__ == "__main__":
  main()
