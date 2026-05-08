#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-./model/Qwen2.5-VL-3B-Instruct}"
DST_DIR="${2:-./saves}"
DRY_RUN="${3:-}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "[ERROR] Source directory not found: $SRC_DIR" >&2
  exit 1
fi

if [[ ! -d "$DST_DIR" ]]; then
  echo "[ERROR] Destination directory not found: $DST_DIR" >&2
  exit 1
fi

# Identify model-weight artifacts and keep them in destination.
# Includes shard files, shard index JSONs, and common adapter (LoRA/PEFT) files.
is_weight_file() {
  local rel_path="$1"
  local base_name
  base_name="$(basename "$rel_path")"

  shopt -s nocasematch
  case "$base_name" in
    *.safetensors|*.pt|*.pth|*.ckpt|*.gguf|*.onnx|*.msgpack)
      shopt -u nocasematch
      return 0
      ;;
    pytorch_model*.bin|adapter_model*.bin|tf_model*.h5)
      shopt -u nocasematch
      return 0
      ;;
    *.data-00000-of-*|*.index|*.safetensors.index.json|*.bin.index.json)
      shopt -u nocasematch
      return 0
      ;;
    adapter_config.json)
      shopt -u nocasematch
      return 0
      ;;
  esac
  shopt -u nocasematch

  return 1
}

log() {
  echo "[INFO] $*"
}

run_or_echo() {
  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "[DRY-RUN] $*"
  else
    eval "$*"
  fi
}

log "Source      : $SRC_DIR"
log "Destination : $DST_DIR"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  log "Mode        : dry-run"
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

preserved_list="$tmp_dir/preserved.list"
replaced_list="$tmp_dir/replaced.list"
removed_list="$tmp_dir/removed.list"

touch "$preserved_list" "$replaced_list" "$removed_list"

# 1) Remove all non-weight files from destination.
while IFS= read -r -d '' dst_file; do
  rel_path="${dst_file#"$DST_DIR"/}"
  if is_weight_file "$rel_path"; then
    printf '%s\n' "$rel_path" >> "$preserved_list"
  else
    printf '%s\n' "$rel_path" >> "$removed_list"
    run_or_echo "rm -f -- \"$dst_file\""
  fi
done < <(find "$DST_DIR" -type f -print0)

# 2) Copy all non-weight files from source to destination.
while IFS= read -r -d '' src_file; do
  rel_path="${src_file#"$SRC_DIR"/}"
  if ! is_weight_file "$rel_path"; then
    printf '%s\n' "$rel_path" >> "$replaced_list"
    dst_path="$DST_DIR/$rel_path"
    dst_parent="$(dirname "$dst_path")"
    run_or_echo "mkdir -p -- \"$dst_parent\""
    run_or_echo "cp -f -- \"$src_file\" \"$dst_path\""
  fi
done < <(find "$SRC_DIR" -type f -print0)

# 3) Clean up empty directories created by deletions.
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[DRY-RUN] find \"$DST_DIR\" -type d -empty -delete"
else
  find "$DST_DIR" -type d -empty -delete
fi

echo
echo "=== REPLACED FILES ==="
if [[ -s "$replaced_list" ]]; then
  sort -u "$replaced_list"
else
  echo "(none)"
fi

echo
echo "=== PRESERVED FILES (WEIGHT-RELATED) ==="
if [[ -s "$preserved_list" ]]; then
  sort -u "$preserved_list"
else
  echo "(none)"
fi

echo
echo "=== REMOVED ONLY (NOT RESTORED FROM SOURCE) ==="
if [[ -s "$removed_list" ]]; then
  if [[ -s "$replaced_list" ]]; then
    grep -Fxv -f "$replaced_list" "$removed_list" | sort -u || true
  else
    sort -u "$removed_list"
  fi
else
  echo "(none)"
fi

log "Done. Non-weight files were replaced while keeping weight files untouched."
