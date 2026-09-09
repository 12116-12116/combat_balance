from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

from .analytics import BatchResult, simulate_batch
from .config import build_scenario, get_by_path, project_with_values
from .models import ProjectConfig


@dataclass
class SensitivityResult:
    baseline: BatchResult
    oat: pd.DataFrame
    correlations: pd.DataFrame
    samples: pd.DataFrame


def analyze_sensitivity(
    project: ProjectConfig,
    parameters: Iterable[str] | None = None,
    runs: int = 300,
    samples: int = 64,
    seed: int = 20260908,
    workers: int = 1,
) -> SensitivityResult:
    paths = list(parameters or [item.path for item in project.optimization.parameters])
    if not paths:
        raise ValueError("至少需要一个敏感性参数")
    base_values = {path: float(get_by_path(project.model_dump(mode="python"), path)) for path in paths}
    scenario = build_scenario(
        project,
        project.study.default_player,
        project.study.default_enemy,
        project.optimization.levels[0],
        project.study.default_policy,
    )
    baseline = simulate_batch(scenario, runs=runs, seed=seed, workers=workers)
    baseline_ttk = baseline.summary["ttk_mean"]
    baseline_win = baseline.summary["win_rate"]

    oat_rows: list[dict[str, Any]] = []
    for path_index, path in enumerate(paths):
        value = base_values[path]
        for factor in (0.9, 0.95, 1.05, 1.1):
            changed = project_with_values(project, {path: value * factor})
            changed_scenario = build_scenario(
                changed,
                changed.study.default_player,
                changed.study.default_enemy,
                changed.optimization.levels[0],
                changed.study.default_policy,
            )
            result = simulate_batch(
                changed_scenario,
                runs=runs,
                seed=seed + path_index * 100 + int(factor * 100),
                workers=workers,
            )
            ttk = result.summary["ttk_mean"]
            win_rate = result.summary["win_rate"]
            relative_change = factor - 1
            ttk_elasticity = None
            if baseline_ttk and ttk is not None:
                ttk_elasticity = ((ttk - baseline_ttk) / baseline_ttk) / relative_change
            win_elasticity = None
            if baseline_win:
                win_elasticity = ((win_rate - baseline_win) / baseline_win) / relative_change
            oat_rows.append({
                "parameter": path,
                "factor": factor,
                "value": value * factor,
                "ttk_mean": ttk,
                "win_rate": win_rate,
                "dps_mean": result.summary["dps_mean"],
                "ttk_elasticity": ttk_elasticity,
                "win_rate_elasticity": win_elasticity,
            })

    bounds_by_path = {item.path: (item.minimum, item.maximum) for item in project.optimization.parameters}
    bounds = [
        bounds_by_path.get(path, (base_values[path] * 0.8, base_values[path] * 1.2))
        for path in paths
    ]
    sampler = qmc.LatinHypercube(d=len(paths), seed=seed + 313)
    unit = sampler.random(n=samples)
    matrix = qmc.scale(unit, [bound[0] for bound in bounds], [bound[1] for bound in bounds])
    sample_rows: list[dict[str, Any]] = []
    sample_runs = max(30, runs // 3)
    for index, values in enumerate(matrix):
        values_by_path = dict(zip(paths, values, strict=True))
        changed = project_with_values(project, values_by_path)
        changed_scenario = build_scenario(
            changed,
            changed.study.default_player,
            changed.study.default_enemy,
            changed.optimization.levels[0],
            changed.study.default_policy,
        )
        result = simulate_batch(changed_scenario, sample_runs, seed + 10000 + index, workers=1)
        row = {**values_by_path}
        row.update({
            "ttk_mean": result.summary["ttk_mean"],
            "win_rate": result.summary["win_rate"],
            "dps_mean": result.summary["dps_mean"],
        })
        sample_rows.append(row)
    sample_frame = pd.DataFrame(sample_rows)

    correlation_rows: list[dict[str, Any]] = []
    for path in paths:
        for metric in ("ttk_mean", "win_rate", "dps_mean"):
            valid = sample_frame[[path, metric]].dropna()
            has_variation = len(valid) >= 3 and valid[path].nunique() > 1 and valid[metric].nunique() > 1
            correlation, p_value = spearmanr(valid[path], valid[metric]) if has_variation else (np.nan, np.nan)
            correlation_rows.append({
                "parameter": path,
                "metric": metric,
                "spearman": float(correlation),
                "p_value": float(p_value),
                "samples": len(valid),
            })
    return SensitivityResult(
        baseline=baseline,
        oat=pd.DataFrame(oat_rows),
        correlations=pd.DataFrame(correlation_rows),
        samples=sample_frame,
    )
