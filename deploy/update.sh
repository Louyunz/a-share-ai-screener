#!/usr/bin/env bash
# 服务器端：定期拉取 GitHub master 并在代码变化时重建容器。
# 安装：见 deploy/README.md；不要用 git pull 覆盖 .env / data（已 gitignore）。
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/a-share-ai-screener}"
BRANCH="${BRANCH:-master}"
LOG="${LOG:-/var/log/a-share-update.log}"

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
# 构建失败时保留旧容器；成功后替换。
docker-compose build --pull
docker-compose up -d
echo "$(date -Is) deployed $after"
