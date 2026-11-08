#!/usr/bin/env bash
# 服务器端：定期拉取 GitHub master（国内通过 ghfast 镜像），有更新时按需装依赖并重启服务。
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/a-share-ai-screener}"
BRANCH="${BRANCH:-master}"
PY="${PY:-$REPO_DIR/.venv/bin/python}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
SERVICE="${SERVICE:-a-share-ai}"

cd "$REPO_DIR"
before="$(git rev-parse HEAD)"
git fetch --quiet origin "$BRANCH"
git pull --ff-only --quiet origin "$BRANCH"
after="$(git rev-parse HEAD)"

if [ "$before" = "$after" ]; then
  echo "$(date -Is) no update ($after)"
  exit 0
fi

echo "$(date -Is) update $before -> $after"
if ! git diff --quiet "$before" "$after" -- requirements.txt; then
  "$PY" -m pip install -r requirements.txt -i "$PIP_INDEX" --disable-pip-version-check
fi
systemctl restart "$SERVICE"
echo "$(date -Is) deployed $after"
