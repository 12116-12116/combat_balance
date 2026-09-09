"""战斗数值模拟器的公共接口。"""

from .analytics import BatchResult, simulate_batch
from .config import build_scenario, load_project
from .engine import BattleResult, simulate_once
from .models import ProjectConfig, ScenarioConfig
from .optimizer import OptimizationResult, optimize_balance
from .reporting import ReportManifest, generate_report
from .sensitivity import SensitivityResult, analyze_sensitivity

__all__ = [
    "BatchResult",
    "BattleResult",
    "OptimizationResult",
    "ProjectConfig",
    "ReportManifest",
    "ScenarioConfig",
    "SensitivityResult",
    "analyze_sensitivity",
    "build_scenario",
    "generate_report",
    "load_project",
    "optimize_balance",
    "simulate_batch",
    "simulate_once",
]

