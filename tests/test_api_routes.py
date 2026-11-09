# -*- coding: utf-8 -*-
"""FastAPI 路由注册检查（不启动 HTTP 服务，不需要 httpx/TestClient）。"""
from __future__ import annotations

import unittest

from backend.main import app


class ApiRouteTest(unittest.TestCase):
    def test_workbench_action_routes_are_registered(self) -> None:
        routes = {}
        for route in app.routes:
            path = getattr(route, "path", "")
            methods = getattr(route, "methods", set()) or set()
            if path:
                routes[path] = set(methods)
        self.assertIn("/api/sessions/{sid}/candidates/{index}/apply", routes)
        self.assertIn("POST", routes["/api/sessions/{sid}/candidates/{index}/apply"])
        self.assertIn("/api/sessions/{sid}/versions/{version_id}/activate", routes)
        self.assertIn("POST", routes["/api/sessions/{sid}/versions/{version_id}/activate"])
        self.assertIn("/api/sessions/{sid}/state", routes)
        self.assertIn("GET", routes["/api/sessions/{sid}/state"])


if __name__ == "__main__":
    unittest.main()
