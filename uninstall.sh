#!/usr/bin/env bash
# Remove what install.sh set up: the systemd user service, the opencode provider and the settings.
# Downloaded models are kept unless you pass --models (they are large and slow to download again).
set -euo pipefail
DATA=${XDG_DATA_HOME:-$HOME/.local/share}/arcangel
CONF_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/arcangel
UNIT=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/arcangel.service
OPENCODE_JSON=${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.json
models=0; [ "${1:-}" = --models ] && models=1

if [ -f "$UNIT" ]; then
  systemctl --user disable --now arcangel 2>/dev/null || true
  rm -f "$UNIT"
  systemctl --user daemon-reload
  echo "removed the arcangel service"
fi
if [ -f "$OPENCODE_JSON" ]; then
  python3 - "$OPENCODE_JSON" <<'PY' && echo "removed the arcangel provider from opencode" || true
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
if cfg.get("provider", {}).pop("arcangel", None) is None:
    sys.exit(1)
json.dump(cfg, open(p, "w"), indent=2)
PY
fi
rm -rf "$CONF_DIR" "$DATA/assets"
echo "removed $CONF_DIR"
if [ $models = 1 ]; then
  rm -rf "$DATA/models"
  echo "removed $DATA/models"
elif [ -d "$DATA/models" ]; then
  echo "kept the models in $DATA/models ($(du -sh "$DATA/models" | cut -f1)); run with --models to delete them"
fi
echo "The source folder (with build/ and server/.venv) is left alone: delete it yourself if you like."
