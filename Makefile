export RENDER_NODE ?= /dev/dri/renderD128

.DEFAULT_GOAL := .env

# Keep an existing local configuration even if the template changes.
.env: | .env.example
	@set -eu; \
	[ ! -e "$@" ] || { printf '%s already exists, leaving it alone\n' "$@" >&2; exit 1; }; \
	node=$$(realpath "$$RENDER_NODE"); \
	[ -c "$$node" ] || { printf 'Not a GPU render device: %s\n' "$$node" >&2; exit 1; }; \
	uid=$$(id -u); gid=$$(id -g); render_gid=$$(stat -c '%g' -- "$$node"); \
	config=$$(RUN_UID="$$uid" RUN_GID="$$gid" RENDER_GID="$$render_gid" RENDER_NODE="$$node" \
		MODELS_DIR="$$HOME/models" \
		awk -F= '$$1 ~ /^(RUN_UID|RUN_GID|RENDER_GID|RENDER_NODE|MODELS_DIR)$$/ { $$0 = $$1 "=" ENVIRON[$$1] } 1' "$|"); \
	printf '%s\n' "$$config" > "$@"
