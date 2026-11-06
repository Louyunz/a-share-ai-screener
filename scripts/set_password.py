# -*- coding: utf-8 -*-
"""生成 AUTH_PASSWORD_HASH，供 .env 配置管理员账号。

用法：
    .\.venv\Scripts\python.exe scripts\set_password.py
    # 也可以指定用户名：... scripts\set_password.py --username admin
密码不会回显，也不会写入命令历史。
"""
from __future__ import annotations

import argparse
import getpass
import sys

from backend.auth import hash_password, validate_password


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 AUTH_USERNAME / AUTH_PASSWORD_HASH")
    parser.add_argument("--username", default="admin", help="管理员用户名（默认 admin）")
    args = parser.parse_args()

    password = getpass.getpass("设置管理员密码：")
    confirm = getpass.getpass("再次输入密码：")
    if password != confirm:
        print("两次输入不一致。")
        return 1
    try:
        validate_password(password)
    except Exception as exc:  # noqa: BLE001
        print(f"密码不符合要求：{exc}")
        return 1

    print("\n把下面两行写入 .env（不要提交到 Git），然后重启服务：")
    print(f"AUTH_USERNAME={args.username}")
    print(f"AUTH_PASSWORD_HASH={hash_password(password)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
