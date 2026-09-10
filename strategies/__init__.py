"""Strategy package. Importing it registers every trader callsign."""

from strategies.base import Strategy, build_strategies, registry
from strategies.orbit import OrbitStrategy
from strategies.flux import FluxStrategy
from strategies.revert import RevertStrategy
from strategies.surge import SurgeStrategy
from strategies.spark import SparkStrategy
from strategies.anchor import AnchorStrategy

__all__ = [
    "Strategy", "build_strategies", "registry",
    "OrbitStrategy", "FluxStrategy", "RevertStrategy",
    "SurgeStrategy", "SparkStrategy", "AnchorStrategy",
]
