import argparse
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import paramiko
from paramiko.ssh_exception import AuthenticationException
from datacrunch import DataCrunchClient
from datacrunch.exceptions import APIException
from dotenv import load_dotenv


# -------------
# Verda helpers
# -------------

def log(msg: str) -> None:
  print(f"[VERDA-MGR] {msg}", flush=True)


def get_verda_client() -> DataCrunchClient:
  try:
    client_id = os.environ["DATACRUNCH_CLIENT_ID"]
    client_secret = os.environ["DATACRUNCH_CLIENT_SECRET"]
  except KeyError as e:
    raise SystemExit(f"Environment variable {e.args[0]} is required for Verda API auth.") from e

  return DataCrunchClient(client_id, client_secret)


@dataclass
class InstanceSpec:
  gpu_model: str
  gpus_per_instance: int
  image: str
  location: str
  ssh_key_name: str
  ssh_private_key: Path
  ssh_user: str
  hostname_prefix: str
  description_prefix: str


def select_instance_type(client: DataCrunchClient, gpu_model: str, gpus_per_instance: int):
  gpu_model_upper = gpu_model.upper()
  types = client.instance_types.get()

  candidates = []
  for t in types:
    itype = t.instance_type  # e.g. "4H100.80S.176V"

    # leading digits = number of GPUs
    digits = []
    for ch in itype:
      if ch.isdigit():
        digits.append(ch)
      else:
        break
    if not digits:
      continue
    try:
      count = int(" ".join(digits).replace(" ", ""))
    except ValueError:
      continue

    if count != gpus_per_instance:
      continue
    if gpu_model_upper not in itype.upper():
      continue
    candidates.append(t)

  if not candidates:
    raise SystemExit(
      f"No instance_type found for gpu_model={gpu_model} with {gpus_per_instance} GPU(s). "
      "Check available instance types on the Verda dashboard."
    )

  candidates.sort(key=lambda t: t.spot_price_per_hour)
  chosen = candidates[0]
  log(
    f"Selected instance_type={chosen.instance_type} "
    f"spot_price=${chosen.spot_price_per_hour:.3f}/h"
  )
  return chosen.instance_type


def get_ssh_key_id(client: DataCrunchClient, key_name: str) -> str:
  keys = client.ssh_keys.get()
  for k in keys:
    if k.name == key_name:
      log(f"Using SSH key '{key_name}' (id={k.id})")
      return k.id
  raise SystemExit(f"SSH key named '{key_name}' not found in your Verda account.")


def create_spot_instance(client: DataCrunchClient, spec: InstanceSpec) -> str:
  instance_type = select_instance_type(client, spec.gpu_model, spec.gpus_per_instance)
  ssh_key_id = get_ssh_key_id(client, spec.ssh_key_name)

  location = choose_location_for_instance_type(
    client=client,
    instance_type=instance_type,
    preferred_location=spec.location,
    is_spot=True,
  )

  hostname = f"{spec.hostname_prefix}-{int(time.time())}"
  description = (
    f"{spec.description_prefix} (gpu={spec.gpu_model}x{spec.gpus_per_instance}, loc={location})"
  )

  log(f"Creating spot instance hostname={hostname} type={instance_type} location={location}")
  inst = client.instances.create(
    instance_type=instance_type,
    image=spec.image,
    hostname=hostname,
    description=description,
    ssh_key_ids=[ssh_key_id],
    location=location,
    is_spot=True,
    contract="SPOT",
  )
  return inst.id


def list_detached_os_volumes(client: DataCrunchClient):
  """Return OS volumes that are detached and not deleted."""
  volumes = client.volumes.get()
  candidates = []
  for v in volumes:
    if getattr(v, "deleted_at", None):
      continue
    if not getattr(v, "is_os_volume", False):
      continue
    if getattr(v, "instance_id", None):
      continue
    candidates.append(v)
  candidates.sort(key=lambda v: getattr(v, "created_at", ""), reverse=True)
  return candidates


def select_os_volume(client: DataCrunchClient, requested_volume_id: Optional[str] = None):
  """Pick an OS volume to reuse (detached / from dead instance)."""
  if requested_volume_id:
    vol = client.volumes.get_by_id(requested_volume_id)
    if not getattr(vol, "is_os_volume", False):
      raise SystemExit(f"Volume {requested_volume_id} is not marked as an OS volume; refusing to boot from it.")
    if getattr(vol, "deleted_at", None):
      raise SystemExit(f"Volume {requested_volume_id} is in trash/deleted state; cannot use it.")
    if getattr(vol, "instance_id", None):
      raise SystemExit(
        f"Volume {requested_volume_id} is still attached to instance {vol.instance_id}. "
        "Detach it from the dashboard first."
      )
    log(f"Reusing specified OS volume {vol.id} (name={vol.name}, loc={vol.location}, size={vol.size}GB).")
    return vol

  candidates = list_detached_os_volumes(client)
  if not candidates:
    raise SystemExit(
      "Detached OS volumes were not found. "
      "If a spot instance was killed, confirm the OS volume is still present and detached in Verda."
    )

  log("Select a detached OS volume to boot from (likely from a killed instance):")
  for idx, v in enumerate(candidates, start=1):
    print(
      f"  [{idx}] id={v.id} name={v.name} size={v.size}GB loc={v.location} "
      f"status={v.status} created_at={v.created_at}"
    )
  while True:
    ans = input(f"Choose volume [1-{len(candidates)}]: ").strip()
    if not ans.isdigit():
      print("Please input a number.")
      continue
    i = int(ans)
    if 1 <= i <= len(candidates):
      vol = candidates[i - 1]
      log(f"Selected volume {vol.id} ({vol.name})")
      return vol
    print("Invalid choice.")


def create_instance_from_os_volume(
  client: DataCrunchClient,
  spec: InstanceSpec,
  os_volume,
) -> str:
  """Create a spot instance that boots from an existing OS volume."""
  instance_type = select_instance_type(client, spec.gpu_model, spec.gpus_per_instance)
  ssh_key_id = get_ssh_key_id(client, spec.ssh_key_name)

  volume_keys = getattr(os_volume, "ssh_key_ids", []) or []
  if volume_keys and ssh_key_id not in volume_keys:
    log(
      f"[WARN] Selected OS volume does not list SSH key '{spec.ssh_key_name}' (id={ssh_key_id}). "
      "Provisioning will still request this key, but ensure it is authorized on the volume."
    )

  volume_location = getattr(os_volume, "location", None)
  if not volume_location:
    raise SystemExit(f"Volume {os_volume.id} has no location info; cannot schedule an instance.")

  if not client.instances.is_available(
    instance_type=instance_type,
    is_spot=True,
    location_code=volume_location,
  ):
    raise SystemExit(
      f"Spot availability for {instance_type} at {volume_location} is currently 0. "
      "Cannot boot from the selected OS volume. Try later or choose a different GPU size."
    )

  hostname = f"{spec.hostname_prefix}-{int(time.time())}"
  description = (
    f"{spec.description_prefix} reuse {os_volume.id[:8]} gpu={spec.gpu_model}x{spec.gpus_per_instance} "
    f"loc={volume_location}"
  )
  if len(description) >= 60:
    description = description[:57] + "..."

  log(
    f"Creating spot instance {instance_type} at {volume_location} from existing OS volume {os_volume.id} "
    f"(size={os_volume.size}GB, name={os_volume.name})"
  )
  inst = client.instances.create(
    instance_type=instance_type,
    image=os_volume.id,
    hostname=hostname,
    description=description,
    ssh_key_ids=[ssh_key_id],
    location=volume_location,
    is_spot=True,
    contract="SPOT",
  )
  return inst.id


def wait_for_instance_ip(client: DataCrunchClient, instance_id: str, timeout_sec: int = 900) -> str:
  deadline = time.time() + timeout_sec
  while time.time() < deadline:
    inst = client.instances.get_by_id(instance_id)
    if getattr(inst, "ip", None):
      log(f"Instance {instance_id} is up with IP {inst.ip}")
      return inst.ip
    log(f"Waiting for IP assignment for instance {instance_id}...")
    time.sleep(15)
  raise TimeoutError(f"Timeout while waiting for instance {instance_id} to get an IP address.")


def delete_instance(client: DataCrunchClient, instance_id: str) -> None:
  log(f"Deleting instance {instance_id} to stop billing...")
  client.instances.action(
    instance_id,
    client.constants.instance_actions.DELETE,
  )
  log("Delete requested.")


KNOWN_LOCATIONS = ["FIN-01", "FIN-02", "FIN-03", "ICE-01"]


def find_available_locations_for_instance_type(
  client: DataCrunchClient,
  instance_type: str,
  is_spot: bool = True,
) -> list[str]:
  """Check each known location for spot availability of a given instance_type."""
  available: list[str] = []
  for loc in KNOWN_LOCATIONS:
    try:
      if client.instances.is_available(
        instance_type=instance_type,
        is_spot=is_spot,
        location_code=loc,
      ):
        available.append(loc)
    except APIException as e:
      # Skip locations that are invalid or temporarily unreachable
      log(f"[DEBUG] is_available failed for {instance_type}@{loc}: {e.code} {e.message}")
      continue
  return available


def choose_location_for_instance_type(
  client: DataCrunchClient,
  instance_type: str,
  preferred_location: Optional[str],
  is_spot: bool = True,
) -> str:
  """
  Determine location for the given instance_type (spot):
  - preferred_location is set and not 'auto': use it if available, otherwise suggest alternatives.
  - preferred_location is 'auto' or None: probe known locations, auto-pick if single, otherwise prompt.
  """
  if preferred_location and preferred_location.lower() != "auto":
    if client.instances.is_available(
      instance_type=instance_type,
      is_spot=is_spot,
      location_code=preferred_location,
    ):
      log(f"Using user-specified location={preferred_location}")
      return preferred_location

    candidates = find_available_locations_for_instance_type(
      client,
      instance_type=instance_type,
      is_spot=is_spot,
    )
    msg_lines = [
      f"Requested location '{preferred_location}' has no spot availability for {instance_type}.",
    ]
    if candidates:
      msg_lines.append("Available locations for this instance type (spot):")
      msg_lines.extend(f"  - {loc}" for loc in candidates)
      msg_lines.append(
        "Re-run with one of the locations above, or omit --location to select interactively."
      )
    else:
      msg_lines.append("No location currently reports spot availability for this instance type.")
    raise SystemExit("\n".join(msg_lines))

  candidates = find_available_locations_for_instance_type(
    client,
    instance_type=instance_type,
    is_spot=is_spot,
  )
  if not candidates:
    raise SystemExit(
      f"No location currently reports spot availability for instance_type={instance_type}. "
      "Try a different GPU model or size."
    )

  if len(candidates) == 1:
    chosen = candidates[0]
    log(f"Auto-selected location={chosen} for instance_type={instance_type}")
    return chosen

  log(f"Available locations for {instance_type} (spot):")
  for idx, loc in enumerate(candidates, start=1):
    print(f"  [{idx}] {loc}")

  while True:
    ans = input(f"Select location [1-{len(candidates)}]: ").strip()
    if not ans.isdigit():
      print("Please input a number.")
      continue
    i = int(ans)
    if 1 <= i <= len(candidates):
      chosen = candidates[i - 1]
      log(f"Selected location={chosen}")
      return chosen
    print("Invalid choice.")


# -------------
# SSH helpers
# -------------

def load_private_key(path: Path):
  last_exc = None
  for key_cls in (paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey):
    try:
      return key_cls.from_private_key_file(str(path))
    except Exception as e:  # noqa: BLE001
      last_exc = e
  raise last_exc  # type: ignore[misc]


def wait_for_ssh(host: str, user: str, pkey, timeout_sec: int = 600) -> paramiko.SSHClient:
  deadline = time.time() + timeout_sec
  while time.time() < deadline:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
      client.connect(hostname=host, username=user, pkey=pkey, timeout=30)
      log(f"SSH connected: {user}@{host}")
      return client
    except AuthenticationException as e:
      raise SystemExit(
        f"SSH authentication failed for {user}@{host}. "
        "Check --ssh-user, --ssh-private-key, and the registered SSH key on Verda.\n"
        f"Details: {e}"
      )
    except Exception as e:  # noqa: BLE001
      log(f"SSH not ready yet ({e!r}), retrying...")
      time.sleep(10)
  raise TimeoutError(f"SSH connection timed out for {user}@{host}")


def sftp_mkdir_p(sftp: paramiko.SFTPClient, remote_path: str) -> None:
  parts = remote_path.strip("/").split("/")
  path = ""
  for p in parts:
    path = f"{path}/{p}" if path else f"/{p}"
    try:
      sftp.stat(path)
    except IOError:
      sftp.mkdir(path)


def sftp_get_recursive(sftp: paramiko.SFTPClient, remote_path: str, local_path: Path) -> None:
  local_path.mkdir(parents=True, exist_ok=True)
  for attr in sftp.listdir_attr(remote_path):
    rname = attr.filename
    rpath = f"{remote_path}/{rname}"
    lpath = local_path / rname
    if paramiko.S_ISDIR(attr.st_mode):
      sftp_get_recursive(sftp, rpath, lpath)
    else:
      sftp.get(rpath, str(lpath))


# -------------
# High-level flow
# -------------

def deploy_remote_payload(
  ssh: paramiko.SSHClient,
  env_file: Path,
  train_config: Path,
  project_root: Path,
  remote_base_dir: str = "~/lerobot_run",
) -> None:
  sftp = ssh.open_sftp()
  try:
    # Expand ~ manually
    stdin, stdout, stderr = ssh.exec_command(f"cd {remote_base_dir} && pwd")
    base_dir_resolved = stdout.read().decode().strip()
    if not base_dir_resolved:
      # Create and re-check
      ssh.exec_command(f"mkdir -p {remote_base_dir}")
      stdin, stdout, stderr = ssh.exec_command(f"cd {remote_base_dir} && pwd")
      base_dir_resolved = stdout.read().decode().strip()
    log(f"Remote base dir: {base_dir_resolved}")

    # Ensure dir exists
    sftp_mkdir_p(sftp, base_dir_resolved)

    # Upload .env -> {base_dir}/.env
    if env_file.is_file():
      sftp.put(str(env_file), f"{base_dir_resolved}/.env")
      log(f"Uploaded env file to {base_dir_resolved}/.env")

    # Upload training config YAML (flattened to base_dir)
    if train_config.is_file():
      remote_cfg = f"{base_dir_resolved}/{train_config.name}"
      sftp.put(str(train_config), remote_cfg)
      log(f"Uploaded training config to {remote_cfg}")

    # Upload remote scripts
    remote_dir_local = project_root / "remote"
    for fname in ("setup_env.sh", "train_lerobot_entry.py", "read_train_config.py"):
      src = remote_dir_local / fname
      dst = f"{base_dir_resolved}/{fname}"
      sftp.put(str(src), dst)
      if fname.endswith(".sh"):
        # make executable
        ssh.exec_command(f"chmod +x {dst}")
      log(f"Uploaded {fname} to {dst}")
  finally:
    sftp.close()


def run_remote_training(
  ssh: paramiko.SSHClient,
  mode: str,
  checkpoint_repo_id: Optional[str] = None,
  train_config_name: Optional[str] = None,
  remote_base_dir: str = "~/lerobot_run",
) -> int:
  # Build command with optional env override for checkpoint repo
  env_prefix = ""
  if checkpoint_repo_id:
    env_prefix = f"LEROBOT_CHECKPOINT_REPO_ID={shlex.quote(checkpoint_repo_id)} "
  if train_config_name:
    env_prefix += f"LEROBOT_TRAIN_CONFIG={shlex.quote(train_config_name)} "

  cmd = f"cd {remote_base_dir} && {env_prefix}bash ./setup_env.sh {mode}"
  log(f"Executing remote command: {cmd}")

  stdin, stdout, stderr = ssh.exec_command(cmd)

  # Stream stdout & stderr
  for line in stdout:
    print(f"[REMOTE-OUT] {line}", end="")
  for line in stderr:
    print(f"[REMOTE-ERR] {line}", end="", file=sys.stderr)

  exit_status = stdout.channel.recv_exit_status()
  log(f"Remote command exited with status {exit_status}")
  return exit_status


def fetch_remote_logs(
  ssh: paramiko.SSHClient,
  run_id: str,
  local_root: Path,
  remote_base_dir: str = "~/lerobot_run",
) -> None:
  sftp = ssh.open_sftp()
  try:
    # Resolve remote_base_dir
    stdin, stdout, stderr = ssh.exec_command(f"cd {remote_base_dir} && pwd")
    base_dir_resolved = stdout.read().decode().strip()
    if not base_dir_resolved:
      log("[WARN] Cannot resolve remote base dir, skipping log download.")
      return

    # Read output_dir.txt written by train_lerobot_entry.py
    output_dir_txt = f"{base_dir_resolved}/output_dir.txt"
    try:
      local_tmp = local_root / f"{run_id}_output_dir.txt"
      sftp.get(output_dir_txt, str(local_tmp))
      output_dir = Path(local_tmp.read_text(encoding="utf-8").strip())
      log(f"Remote training output_dir (from output_dir.txt): {output_dir}")
    except IOError:
      log("[WARN] output_dir.txt not found on remote; skipping structured log download.")
      return

    # Download the whole output_dir tree
    remote_output_dir = str(output_dir)
    local_run_dir = local_root / run_id / "outputs"
    log(f"Downloading remote outputs from {remote_output_dir} to {local_run_dir}")
    sftp_get_recursive(sftp, remote_output_dir, local_run_dir)

    # Also grab setup_env logs
    for log_name in ("setup_env_train.log", "setup_env_resume_hub.log", "setup_env_resume_local.log"):
      remote_log = f"{base_dir_resolved}/{log_name}"
      local_log = local_root / run_id / log_name
      try:
        local_log.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_log, str(local_log))
      except IOError:
        # ignore missing logs
        continue

    log(f"Log download for run_id={run_id} complete.")
  finally:
    sftp.close()


# -------------
# CLI
# -------------

def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="End-to-end LeRobot training manager on Verda spot instances",
  )
  sub = p.add_subparsers(dest="command", required=True)

  def add_common(sp, *, include_image: bool = True, include_location: bool = True):
    sp.add_argument("--gpu-model", required=True, help="GPU model, e.g. H100, A100, L40S")
    sp.add_argument("--gpus-per-instance", type=int, required=True, help="GPUs per instance")
    if include_image:
      sp.add_argument(
        "--image",
        default="ubuntu-24.04-cuda-12.8-open-docker",
        help="Verda image name (ignored for resume_local because an existing OS volume is used)",
      )
    if include_location:
      sp.add_argument(
        "--location",
        default="auto",
        help=(
          "Verda location code (e.g. FIN-01, FIN-02, FIN-03, ICE-01). "
          "Use 'auto' to probe available locations interactively."
        ),
      )
    sp.add_argument(
      "--env-file",
      default=".env",
      help="Local env file containing secrets to upload to the remote instance",
    )
    sp.add_argument(
      "--train-config",
      default="configs/train.remote.yaml",
      help="Training config YAML to upload alongside the remote scripts",
    )
    sp.add_argument(
      "--hostname-prefix",
      default="lerobot",
      help="Prefix for instance hostname",
    )
    sp.add_argument(
      "--description-prefix",
      default="lerobot-train",
      help="Prefix for instance description",
    )
    sp.add_argument("--ssh-user", default="root", help="SSH username (default: root)")
    sp.add_argument(
      "--run-id",
      default=None,
      help="Optional run id for local log directory naming; default is timestamp-based",
    )

  train_p = sub.add_parser("train", help="Start a new training run (or local resume)")
  add_common(train_p)

  resume_p = sub.add_parser("resume", help="Resume training from a checkpoint repo on Hugging Face")
  add_common(resume_p)
  resume_p.add_argument(
    "--checkpoint-repo-id",
    required=True,
    help="Hugging Face model repo_id that stores the last checkpoint "
         "(uploaded by previous runs).",
  )

  resume_local_p = sub.add_parser(
    "resume_local",
    help="Resume training by booting from an existing OS volume (detached storage) and local checkpoints",
  )
  add_common(resume_local_p, include_image=False, include_location=False)
  resume_local_p.add_argument(
    "--os-volume-id",
    help="Existing OS volume ID to boot from. If omitted, you will be prompted to choose a detached OS volume.",
  )
  resume_local_p.add_argument(
    "--checkpoint-repo-id",
    required=False,
    help="Optional HF model repo_id to upload refreshed checkpoints after resume_local",
  )

  return p


def main() -> None:
  parser = build_parser()
  args = parser.parse_args()

  # Load secrets (no override) so CLI args can read them
  load_dotenv(".env", override=False)

  project_root = Path(__file__).resolve().parents[1]
  env_file = Path(args.env_file).expanduser()
  if not env_file.is_file():
    raise SystemExit(f"env file not found: {env_file}")
  if env_file.resolve() != (project_root / ".env").resolve():
    load_dotenv(env_file, override=False)

  train_config = Path(args.train_config).expanduser()
  if not train_config.is_file():
    raise SystemExit(f"training config YAML not found: {train_config}")

  ssh_key_name = os.environ["VERDA_SSH_KEY_NAME"]
  ssh_private_key = os.environ["VERDA_SSH_PRIVATE_KEY"]

  image = getattr(args, "image", "ubuntu-24.04-cuda-12.8-open-docker")
  location = getattr(args, "location", "auto")
  spec = InstanceSpec(
    gpu_model=args.gpu_model,
    gpus_per_instance=args.gpus_per_instance,
    image=image,
    location=location,
    ssh_key_name=ssh_key_name,
    ssh_private_key=Path(ssh_private_key).expanduser(),
    ssh_user=args.ssh_user,
    hostname_prefix=args.hostname_prefix,
    description_prefix=args.description_prefix,
  )

  run_id = args.run_id or str(int(time.time()))
  local_logs_root = Path("runs").resolve()

  client = get_verda_client()
  if args.command == "resume_local":
    volume = select_os_volume(client, getattr(args, "os_volume_id", None))
    spec.image = volume.id
    spec.location = volume.location
    instance_id = create_instance_from_os_volume(client, spec, volume)
  else:
    instance_id = create_spot_instance(client, spec)

  try:
    ip = wait_for_instance_ip(client, instance_id)
    pkey = load_private_key(spec.ssh_private_key)
    ssh = wait_for_ssh(ip, spec.ssh_user, pkey)
    try:
      deploy_remote_payload(ssh, env_file, train_config, project_root)

      if args.command == "train":
        mode = "train"
        checkpoint_repo_id = None
      elif args.command == "resume":
        mode = "resume_hub"
        checkpoint_repo_id = args.checkpoint_repo_id
      else:
        mode = "resume_local"
        checkpoint_repo_id = getattr(args, "checkpoint_repo_id", None)

      status = run_remote_training(
        ssh,
        mode=mode,
        checkpoint_repo_id=checkpoint_repo_id,
        train_config_name=train_config.name,
      )

      if status == 0:
        # Only attempt log download on clean exit
        fetch_remote_logs(ssh, run_id=run_id, local_root=local_logs_root)
      else:
        log(f"Remote training exited with non-zero status ({status}); "
            "skipping log download.")
    finally:
      ssh.close()
  finally:
    # Always attempt to delete the instance to stop billing
    try:
      delete_instance(client, instance_id)
    except Exception as e:  # noqa: BLE001
      log(f"[WARN] Failed to delete instance {instance_id}: {e!r}")


if __name__ == "__main__":
  main()
