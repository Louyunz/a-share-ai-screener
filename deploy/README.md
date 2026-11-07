# 云服务器部署说明（Docker + Nginx）

## 目录

- `Dockerfile`：Python 3.12 运行环境；
- `docker-compose.yml`：应用容器，只绑定 `127.0.0.1:8000`；
- `deploy/update.sh`：拉 GitHub master 并重建容器；
- Nginx 负责 HTTPS 和反向代理（证书路径按服务器实际修改）。

## 首次部署

```bash
git clone https://github.com/Louyunz/a-share-ai-screener.git /opt/a-share-ai-screener
cd /opt/a-share-ai-screener
# 把生产 .env 放到此目录（不要提交）；data/ 放缓存和 DB
chown -R 10001:10001 data
docker-compose build --pull
docker-compose up -d
```

Nginx 反向代理到 `http://127.0.0.1:8000`，必须传递：

```nginx
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto https;
proxy_buffering off;
proxy_read_timeout 3600;
```

## 定期拉最新版

```bash
chmod +x /opt/a-share-ai-screener/deploy/update.sh
cat >/etc/cron.d/a-share-update <<'EOF'
*/5 * * * * root /opt/a-share-ai-screener/deploy/update.sh >> /var/log/a-share-update.log 2>&1
EOF
```

> 自动更新会把公开 GitHub master 的代码部署到线上；确保仓库只有可信任的提交者。
> `.env`、`data/` 已被 `.gitignore` 忽略，`git pull` 不会覆盖生产密钥和数据库。
