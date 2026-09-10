"""Configuration loading for Meridian Capital.

One YAML file is the single source of truth for every tunable parameter.
`Config` wraps it with dotted-path lookup so call sites read cleanly:

    cfg = load_config()
    cfg.get("risk.max_portfolio_drawdown")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

_MISSING = object()


class ConfigError(Exception):
    """Raised when the config file is absent, malformed, or missing a key."""


class Config:
    def __init__(self, data: dict, source: Path | None = None):
        self._data = data
        self.source = source

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Fetch a value by dotted path. Raises if absent and no default given."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"Missing config key: {path}")
                return default
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def section(self, path: str) -> dict:
        value = self.get(path)
        if not isinstance(value, dict):
            raise ConfigError(f"Config key {path} is not a section")
        return value

    @property
    def universe(self) -> list[str]:
        """Flat list of every tradable symbol, stocks first."""
        return list(self.get("universe.stocks")) + list(self.get("universe.crypto"))

    def asset_class(self, symbol: str) -> str:
        """'crypto' or 'stocks' — drives cost model and calendar assumptions."""
        return "crypto" if symbol in self.get("universe.crypto") else "stocks"

    def repo_path(self, relative: str) -> Path:
        return REPO_ROOT / relative

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"<Config {self.get('system.name')} v{self.get('system.version')}>"


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {cfg_path} did not parse to a mapping")
    cfg = Config(data, source=cfg_path)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Fail fast on the handful of settings that would silently corrupt results."""
    required = [
        "universe.stocks",
        "universe.crypto",
        "capital.starting_paper_capital",
        "costs.stocks.commission_bps",
        "costs.crypto.commission_bps",
        "validation.folds",
        "validation.min_folds_passing",
        "strategies",
    ]
    for key in required:
        cfg.get(key)

    folds = cfg.get("validation.folds")
    passing = cfg.get("validation.min_folds_passing")
    if not 1 <= passing <= folds:
        raise ConfigError(
            f"validation.min_folds_passing ({passing}) must be between 1 and folds ({folds})"
        )
    if not 0 < cfg.get("risk.max_portfolio_drawdown") < 1:
        raise ConfigError("risk.max_portfolio_drawdown must be a fraction between 0 and 1")
    if cfg.get("capital.starting_paper_capital") <= 0:
        raise ConfigError("capital.starting_paper_capital must be positive")
