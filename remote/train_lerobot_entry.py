import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from huggingface_hub import HfApi, snapshot_download, upload_folder


def log(msg: str) -> None:
  print(f"[LE-REMOTE] {msg}", flush=True)


def env_bool(name: str, default: bool = False) -> bool:
  v = os.getenv(name)
  if v is None:
    return default
  v = v.strip().lower()
  return v in ("1", "true", "yes", "y", "on")


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


def compute_output_dir(dataset_repo_id: str, policy_type: str) -> str:
  safe = dataset_repo_id.replace("/", "__").replace(".", "_")
  return f"outputs/train/{policy_type}_{safe}"


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


def lerobot_train(mode: str) -> None:
  ensure_hf_tokens()

  dataset_repo_id = os.environ.get("LEROBOT_DATASET_REPO_ID")
  if not dataset_repo_id:
    raise RuntimeError("LEROBOT_DATASET_REPO_ID is required.")

  policy_type = os.environ.get("LEROBOT_POLICY_TYPE", "act")
  policy_path = os.environ.get("LEROBOT_POLICY_PATH")
  device = os.environ.get("LEROBOT_DEVICE", "cuda")
  wandb_enable = env_bool("LEROBOT_WANDB_ENABLE", default=bool(os.getenv("WANDB_API_KEY")))
  steps = os.environ.get("LEROBOT_STEPS")
  batch_size = os.environ.get("LEROBOT_BATCH_SIZE")
  rename_map = os.environ.get("LEROBOT_RENAME_MAP")
  policy_empty_cameras = os.environ.get("LEROBOT_POLICY_EMPTY_CAMERAS")

  # policy upload behaviour
  policy_repo_id = os.environ.get("LEROBOT_POLICY_REPO_ID")

  # optional checkpoint repo for persistence
  checkpoint_repo_id = os.environ.get("LEROBOT_CHECKPOINT_REPO_ID")

  # Decide output_dir and job_name
  output_dir = os.environ.get("LEROBOT_OUTPUT_DIR")
  if not output_dir:
    output_dir = compute_output_dir(dataset_repo_id, policy_type)

  job_name = os.environ.get("LEROBOT_JOB_NAME")
  if not job_name:
    job_name = f"{policy_type}_{dataset_repo_id.split('/')[-1]}"

  # Persist output_dir for the orchestrator
  write_output_dir_file(output_dir)

  if mode == "resume_hub":
    # Resume from a checkpoint stored on the Hub
    checkpoint_repo_id_env = os.environ.get("LEROBOT_CHECKPOINT_REPO_ID")
    if not checkpoint_repo_id_env:
      raise RuntimeError(
        "LEROBOT_CHECKPOINT_REPO_ID must be set in the environment for resume_hub mode."
      )

    log(f"Resuming from Hub checkpoint repo: {checkpoint_repo_id_env}")
    restored_output_dir, config_path_dir = prepare_output_dir_from_hub_checkpoint(
      checkpoint_repo_id_env
    )
    # Overwrite output_dir so we are consistent
    output_dir = restored_output_dir
    write_output_dir_file(output_dir)

    cmd = [
      sys.executable,
      "-m",
      "lerobot.scripts.lerobot_train",
      f"--config_path={config_path_dir}",
      "--resume=true",
    ]
    if rename_map:
      cmd.append(f"--rename_map={rename_map}")
    if policy_empty_cameras:
      cmd.append(f"--policy.empty_cameras={policy_empty_cameras}")
    if steps:
      cmd.append(f"--steps={steps}")

    run_cmd(cmd)

    # After a successful resume run, push the new last checkpoint
    upload_last_checkpoint(output_dir, checkpoint_repo_id_env)
    return

  # Default: "train" mode (fresh or local resume)
  last_cfg = Path(output_dir) / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
  if last_cfg.is_file():
    log(f"Existing local checkpoint found at {last_cfg}, resuming locally.")
    config_path_dir = str(last_cfg.parent)
    cmd = [
      sys.executable,
      "-m",
      "lerobot.scripts.lerobot_train",
      f"--config_path={config_path_dir}",
      "--resume=true",
    ]
    if steps:
      cmd.append(f"--steps={steps}")
  else:
    log("No local checkpoint found, starting a new training run.")
    cmd = [
      sys.executable,
      "-m",
      "lerobot.scripts.lerobot_train",
      f"--dataset.repo_id={dataset_repo_id}",
      f"--output_dir={output_dir}",
      f"--job_name={job_name}",
      f"--policy.device={device}",
      f"--wandb.enable={'true' if wandb_enable else 'false'}",
    ]
    if batch_size:
      cmd.append(f"--batch_size={batch_size}")
    if rename_map:
      cmd.append(f"--rename_map={rename_map}")
    if policy_empty_cameras:
      cmd.append(f"--policy.empty_cameras={policy_empty_cameras}")
    if policy_path:
      cmd.append(f"--policy.path={policy_path}")
    else:
      cmd.append(f"--policy.type={policy_type}")
    if policy_repo_id:
      cmd.append(f"--policy.repo_id={policy_repo_id}")
    else:
      # Avoid ValueError: 'policy.repo_id' argument missing when push_to_hub is true by default
      cmd.append("--policy.push_to_hub=false")

    if steps:
      cmd.append(f"--steps={steps}")

  run_cmd(cmd)

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
  args = parser.parse_args()
  lerobot_train(args.mode)


if __name__ == "__main__":
  main()
