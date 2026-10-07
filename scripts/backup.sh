#!/usr/bin/env bash
# Back up the state that is expensive to recompute to a different filesystem (default: ~/driftguard_backup on the
# ext4 root partition, away from the NTFS project drive). Safe to run while jobs are running:
#   - mlflow.db is copied with SQLite's online backup API (a consistent snapshot, never a half-written file)
#   - mlruns/ (run artifacts), models/ (bundles), data/splits/ (frozen splits) are rsync'd incrementally
# Not backed up: data/raw (re-downloadable, checksums in data/MANIFEST.csv) and data/interim|processed
# (rebuilt from raw in ~1 h with make ingest tracks); code and reports/ live in git.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-$HOME/driftguard_backup}"
mkdir -p "$DEST"
stamp=$(date +%Y%m%d-%H%M)
~/.venvs/driftguard/bin/python - "$DEST/mlflow-$stamp.db" <<'PY'
import sqlite3, sys
src = sqlite3.connect("mlflow.db"); dst = sqlite3.connect(sys.argv[1])
src.backup(dst); dst.close(); src.close()
PY
ln -sfn "mlflow-$stamp.db" "$DEST/mlflow-latest.db"
ls -1t "$DEST"/mlflow-2*.db | tail -n +8 | xargs -r rm --        # keep the 7 newest snapshots
for d in mlruns models data/splits; do
  mkdir -p "$DEST/$d"
  rsync -a --delete "$d/" "$DEST/$d/"
done
cp -f reports/tables/splits_lock.csv data/MANIFEST.csv "$DEST/" 2>/dev/null || true
echo "backup -> $DEST ($(du -sh "$DEST" | cut -f1)); restore: see the header of scripts/backup.sh"
# Restore after a crash: cp $DEST/mlflow-latest.db mlflow.db; rsync -a $DEST/mlruns/ mlruns/;
#   rsync -a $DEST/models/ models/; rsync -a $DEST/data/splits/ data/splits/; make check-splits
