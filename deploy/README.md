# 云服务器部署说明（国内 Python venv + Nginx）

当前生产部署使用 **Python venv + systemd + Nginx**，不依赖 Docker 镜像拉取。
Docker 文件仍保留在仓库中作为备选；Docker Hub 拉取较慢时不要阻塞部署。

## 首次部署

```bash
git clone https://ghfast.top/https://github.com/Louyunz/a-share-ai-screener.git /opt/a-share-ai-screener
cd /opt/a-share-ai-screener
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 把生产 .env、data/app.db、data/a_share 放好
useradd --system --uid 10001 --home-dir /opt/a-share-ai-screener --shell /usr/sbin/nologin ashare
chown -R ashare:ashare data
cp deploy/a-share-ai.service /etc/systemd/system/a-share-ai.service
systemctl daemon-reload
systemctl enable --now a-share-ai
```

Nginx 反向代理到 `http://127.0.0.1:8000`，`deploy/nginx-ip.conf` 是 IP + 自签名证书模板；
必须传递 `X-Forwarded-Proto: https`，否则服务端 Cookie 不会启用 Secure。

## 定期拉取 GitHub 最新版

服务器访问 GitHub 直连较慢，仓库 remote 使用可用镜像：

```bash
git -C /opt/a-share-ai-screener remote set-url origin \
  https://ghfast.top/https://github.com/Louyunz/a-share-ai-screener.git
```

安装 `/etc/cron.d/a-share-update`：

```cron
*/5 * * * * root /opt/a-share-ai-screener/deploy/update.sh >> /var/log/a-share-update.log 2>&1
```

`deploy/update.sh` 只在 HEAD 变化时执行：requirements 有变化才 pip install，然后重启 systemd。
`.env`、`data/` 已被 gitignore，更新不会覆盖生产密钥与数据库。

> 自动更新会部署 GitHub master 的代码；确保仓库只有可信任的提交者。
> 若镜像失效，可把 remote 换成其他可用 GitHub 加速代理，或直连 GitHub。
