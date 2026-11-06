# -*- coding: utf-8 -*-
"""运行配置：路径、LLM、数据与股票池参数。

所有配置均可通过环境变量覆盖；也支持项目根目录下的 .env 文件（KEY=VALUE）。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 读取：仅支持 KEY=VALUE 与 # 注释，不覆盖已有环境变量。"""
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
A_SHARE_DIR = DATA_DIR / "a_share"
DB_PATH = DATA_DIR / "app.db"

# LLM（OpenAI 兼容接口）
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"

# 数据与回测
SOURCE = os.environ.get("SOURCE", "sample")
DEFAULT_UNIVERSE = os.environ.get("UNIVERSE", "csi800")
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", "0.02"))
DEFAULT_TOP_N = int(os.environ.get("DEFAULT_TOP_N", "10"))
BACKTEST_YEARS = float(os.environ.get("BACKTEST_YEARS", "2"))
REBALANCE_DAYS = int(os.environ.get("REBALANCE_DAYS", "20"))
COST_BUY = float(os.environ.get("COST_BUY", "0.0003"))
COST_SELL = float(os.environ.get("COST_SELL", "0.0013"))

# 每日行情同步（收盘后自动拉取）
DATA_SYNC_ENABLED = os.environ.get("DATA_SYNC_ENABLED", "1").strip().lower() not in ("0", "false", "no")
DATA_SYNC_HOUR = int(os.environ.get("DATA_SYNC_HOUR", "18"))
DATA_SYNC_MINUTE = int(os.environ.get("DATA_SYNC_MINUTE", "5"))
DATA_SYNC_SOURCE = os.environ.get("DATA_SYNC_SOURCE", SOURCE)
DATA_SYNC_TIMEZONE = os.environ.get("DATA_SYNC_TIMEZONE", "Asia/Shanghai")

# 管理员登录（单用户）
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "").strip()
AUTH_PASSWORD_HASH = os.environ.get("AUTH_PASSWORD_HASH", "").strip()
AUTH_SESSION_HOURS = float(os.environ.get("AUTH_SESSION_HOURS", "12"))
AUTH_COOKIE_NAME = os.environ.get("AUTH_COOKIE_NAME", "zhitou_admin_session")
AUTH_COOKIE_SECURE = os.environ.get("AUTH_COOKIE_SECURE", "0").strip().lower() in ("1", "true", "yes")
AUTH_ALLOW_SETUP = os.environ.get("AUTH_ALLOW_SETUP", "1").strip().lower() not in ("0", "false", "no")

# 演示账户（单用户）
DEMO_USER = "demo"


def llm_api_key() -> str:
    """读取 LLM API Key（若配置在环境变量中）。"""
    return os.environ.get(LLM_API_KEY_ENV, "") or os.environ.get("OPENAI_API_KEY", "")


class ConfigError(RuntimeError):
    """配置缺失或非法。"""
