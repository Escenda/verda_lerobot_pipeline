import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(f"Config not found: {path}")
  data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
  if not isinstance(data, dict):
    raise ValueError("YAML のトップレベルは mapping である必要があります。")
  return data


def cfg_get(cfg: Dict[str, Any], dotted_key: str):
  cur: Any = cfg
  for part in dotted_key.split("."):
    if not isinstance(cur, dict) or part not in cur:
      return None
    cur = cur[part]
  return cur


def coerce_bool(val: Any):
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


def to_str(val: Any, kind: str) -> str:
  if val is None:
    return ""
  if kind == "raw":
    return json.dumps(val, ensure_ascii=False)
  if kind == "str":
    return str(val)
  if kind == "int":
    try:
      return str(int(val))
    except Exception:
      raise ValueError(f"整数に変換できません: {val!r}")
  if kind == "bool":
    b = coerce_bool(val)
    if b is None:
      raise ValueError(f"真偽値に変換できません: {val!r}")
    return "true" if b else "false"
  raise ValueError(f"Unsupported type: {kind}")


def main() -> None:
  parser = argparse.ArgumentParser(description="YAML から設定値を取得して標準出力へ返します。")
  parser.add_argument("--config", default="train.remote.yaml", help="YAML ファイルパス")
  parser.add_argument("--get", required=True, help="dotted.key 形式で取得したいキー")
  parser.add_argument(
    "--type",
    choices=["raw", "str", "int", "bool"],
    default="str",
    help="出力の型（bool は true/false を出力）",
  )
  parser.add_argument(
    "--default",
    default="",
    help="キーが存在しない場合に出力する値（空文字なら何も出力しません）",
  )
  args = parser.parse_args()

  cfg_path = Path(args.config)
  try:
    cfg = load_yaml(cfg_path)
    val = cfg_get(cfg, args.get)
    if val is None:
      print(args.default, end="")
      return
    print(to_str(val, args.type), end="")
  except Exception as exc:  # noqa: BLE001
    print(f"[read_train_config] {exc}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
