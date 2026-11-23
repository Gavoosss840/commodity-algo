"""Configuration loading (settings.yaml + .env secrets)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional dependency
        return
    load_dotenv(root / ".env")


@dataclass
class Settings:
    raw: dict[str, Any]
    root: Path = field(default=PROJECT_ROOT)

    # --- convenience accessors -------------------------------------------
    @property
    def account(self) -> dict:
        return self.raw["account"]

    @property
    def ib(self) -> dict:
        return self.raw["ib"]

    @property
    def risk(self) -> dict:
        return self.raw["risk"]

    @property
    def signal(self) -> dict:
        return self.raw["signal"]

    @property
    def model(self) -> dict:
        return self.raw["model"]

    @property
    def markets(self) -> dict[str, dict]:
        return self.raw["markets"]

    def enabled_markets(self) -> dict[str, dict]:
        return {k: v for k, v in self.markets.items() if v.get("enabled")}

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    # --- secrets (never stored in yaml) -----------------------------------
    @staticmethod
    def secret(name: str) -> str | None:
        v = os.environ.get(name)
        return v.strip() if v else None


def load_settings(path: str | Path | None = None, root: Path | None = None) -> Settings:
    if root is None:
        # a config file at <project>/config/settings.yaml defines the project root
        root = Path(path).resolve().parent.parent if path else PROJECT_ROOT
    root = Path(root)
    _load_dotenv(root)
    cfg_path = Path(path) if path else root / "config" / "settings.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings(raw=raw, root=root)
