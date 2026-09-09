from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc

from .analytics import simulate_batch
from .config import build_scenario, get_by_path, project_with_values
from .models import OptimizationParameter, ProjectConfig, TargetRange


@dataclass
class OptimizationResult:
    trials: pd.DataFrame
    pareto: pd.DataFrame
    recommended: dict[str, Any]
    validation: pd.DataFrame


def _range_penalty(value: float | None, minimum: float, maximum: float) -> float:
    if value is None or not np.isfinite(value):
        return 3.0
    width = max(maximum - minimum, 1e-9)
    if value < minimum:
        return (minimum - value) / width
    if value > maximum:
        return (value - maximum) / width
    return 0.0


def _candidate_metrics(
    project: ProjectConfig,
    values: dict[str, float],
    runs: int,
    seed: int,
    index: int,
) -> dict[str, Any]:
    candidate = project_with_values(project, values)
    targets = {target.enemy_id: target for target in candidate.study.targets}
    scenarios = [
        (player_id, enemy_id, level)
        for level in candidate.optimization.levels
        for enemy_id in targets
        for player_id in candidate.players
    ]
    runs_per_scenario = max(5, runs // max(1, len(scenarios)))
    rows: list[dict[str, Any]] = []
    for scenario_index, (player_id, enemy_id, level) in enumerate(scenarios):
        scenario = build_scenario(candidate, player_id, enemy_id, level, candidate.players[player_id].default_policy)
        result = simulate_batch(
            scenario,
            runs=runs_per_scenario,
            seed=seed + index * 100003 + scenario_index,
            workers=1,
        )
        rows.append({
            "player": player_id,
            "enemy": enemy_id,
            "level": level,
            "ttk": result.summary["ttk_mean"],
            "win_rate": result.summary["win_rate"],
            "dps": result.summary["dps_mean"],
        })

    frame = pd.DataFrame(rows)
    ttk_penalties: list[float] = []
    win_penalties: list[float] = []
    spreads: list[float] = []
    for (enemy_id, _), group in frame.groupby(["enemy", "level"]):
        target = targets[enemy_id]
        ttk_penalties.append(_range_penalty(group["ttk"].mean(), target.ttk_min, target.ttk_max))
        win_penalties.append(_range_penalty(group["win_rate"].mean(), target.win_rate_min, target.win_rate_max))
        mean_dps = group["dps"].mean()
        spreads.append((group["dps"].max() - group["dps"].min()) / mean_dps if mean_dps else 1.0)
    parameters = candidate.optimization.parameters
    base_dump = project.model_dump(mode="python")
    changes = [
        abs(values[item.path] - float(get_by_path(base_dump, item.path))) / (item.maximum - item.minimum)
        for item in parameters
    ]
    ttk_objective = float(np.mean(ttk_penalties))
    win_objective = float(np.mean(win_penalties))
    class_spread = float(np.mean(spreads)) if spreads else 1.0
    class_objective = max(0.0, class_spread - 0.08) / 0.08
    change_objective = float(np.mean(changes)) if changes else 0.0
    feasible = ttk_objective == 0 and win_objective == 0 and class_spread <= 0.08
    return {
        "candidate": index,
        **values,
        "ttk_objective": ttk_objective,
        "win_objective": win_objective,
        "class_spread": class_spread,
        "class_objective": class_objective,
        "change_objective": change_objective,
        "feasible": feasible,
        "score": 0.35 * ttk_objective + 0.30 * win_objective + 0.20 * class_objective + 0.15 * change_objective,
    }


def _pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    columns = ["ttk_objective", "win_objective", "class_objective", "change_objective"]
    values = frame[columns].to_numpy(dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for index, value in enumerate(values):
        if not keep[index]:
            continue
        dominated = np.all(values <= value, axis=1) & np.any(values < value, axis=1)
        if np.any(dominated):
            keep[index] = False
    return keep


def _sample_lhs(parameters: list[OptimizationParameter], count: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=len(parameters), seed=seed)
    return qmc.scale(
        sampler.random(count),
        [item.minimum for item in parameters],
        [item.maximum for item in parameters],
    )


def _evaluate_matrix(
    project: ProjectConfig,
    matrix: np.ndarray,
    runs: int,
    seed: int,
    start_index: int,
    workers: int,
) -> list[dict[str, Any]]:
    parameters = project.optimization.parameters
    jobs = [
        (
            dict(zip((item.path for item in parameters), values, strict=True)),
            start_index + offset,
        )
        for offset, values in enumerate(matrix)
    ]
    if workers == 1:
        return [_candidate_metrics(project, values, runs, seed, index) for values, index in jobs]
    return Parallel(n_jobs=workers, backend="loky")(
        delayed(_candidate_metrics)(project, values, runs, seed, index) for values, index in jobs
    )


def _validation(
    project: ProjectConfig,
    recommended_values: dict[str, float],
    runs: int,
    seed: int,
    workers: int,
) -> pd.DataFrame:
    candidate = project_with_values(project, recommended_values)
    targets = {item.enemy_id: item for item in candidate.study.targets}
    scenarios = [
        (player, enemy, level)
        for level in candidate.optimization.levels
        for enemy in targets
        for player in candidate.players
    ]
    runs_per_scenario = max(10, runs // len(scenarios))
    rows = []
    for index, (player, enemy, level) in enumerate(scenarios):
        scenario = build_scenario(candidate, player, enemy, level, candidate.players[player].default_policy)
        result = simulate_batch(scenario, runs_per_scenario, seed + 800000 + index, workers=workers)
        target: TargetRange = targets[enemy]
        rows.append({
            "scope": "class",
            "player": player,
            "enemy": enemy,
            "level": level,
            "runs": runs_per_scenario,
            "ttk_mean": result.summary["ttk_mean"],
            "ttk_target": f"{target.ttk_min}-{target.ttk_max}",
            "win_rate": result.summary["win_rate"],
            "dps_mean": result.summary["dps_mean"],
            "dps_spread": None,
            "win_target": f"{target.win_rate_min:.0%}-{target.win_rate_max:.0%}",
            "target_pass": None,
        })
    class_frame = pd.DataFrame(rows)
    aggregate_rows = []
    for (enemy, level), group in class_frame.groupby(["enemy", "level"]):
        target = targets[enemy]
        ttk_mean = group["ttk_mean"].dropna().mean()
        win_rate = group["win_rate"].mean()
        dps_mean = group["dps_mean"].mean()
        dps_spread = (group["dps_mean"].max() - group["dps_mean"].min()) / dps_mean if dps_mean else 1.0
        passed = (
            target.ttk_min <= ttk_mean <= target.ttk_max
            and target.win_rate_min <= win_rate <= target.win_rate_max
            and dps_spread <= 0.08
        )
        aggregate_rows.append({
            "scope": "roster",
            "player": "阵容均值",
            "enemy": enemy,
            "level": level,
            "runs": int(group["runs"].sum()),
            "ttk_mean": ttk_mean,
            "ttk_target": f"{target.ttk_min}-{target.ttk_max}",
            "win_rate": win_rate,
            "dps_mean": dps_mean,
            "dps_spread": dps_spread,
            "win_target": f"{target.win_rate_min:.0%}-{target.win_rate_max:.0%}",
            "target_pass": passed,
        })
    return pd.concat([class_frame, pd.DataFrame(aggregate_rows)], ignore_index=True)


def _validation_score(frame: pd.DataFrame, targets: dict[str, TargetRange]) -> tuple[bool, float]:
    aggregates = frame[frame["scope"] == "roster"]
    penalties = []
    for _, row in aggregates.iterrows():
        target = targets[row["enemy"]]
        penalties.extend([
            _range_penalty(row["ttk_mean"], target.ttk_min, target.ttk_max),
            _range_penalty(row["win_rate"], target.win_rate_min, target.win_rate_max),
            max(0.0, float(row["dps_spread"]) - 0.08) / 0.08,
        ])
    return bool(aggregates["target_pass"].all()), float(np.mean(penalties))


def optimize_balance(
    project: ProjectConfig,
    search_space: list[OptimizationParameter] | None = None,
    seed: int = 20260908,
    workers: int = 1,
    stage1_candidates: int | None = None,
    stage1_runs: int | None = None,
    stage2_candidates: int | None = None,
    stage2_runs: int | None = None,
    final_runs: int | None = None,
) -> OptimizationResult:
    if search_space is not None:
        project = project.model_copy(deep=True)
        project.optimization.parameters = search_space
    parameters = project.optimization.parameters
    if not parameters:
        raise ValueError("优化搜索空间为空")
    count1 = stage1_candidates or project.optimization.stage1_candidates
    runs1 = stage1_runs or project.optimization.stage1_runs
    count2 = stage2_candidates or project.optimization.stage2_candidates
    runs2 = stage2_runs or project.optimization.stage2_runs
    validation_runs = final_runs or project.optimization.final_runs

    base_dump = project.model_dump(mode="python")
    baseline = np.asarray([[float(get_by_path(base_dump, item.path)) for item in parameters]])
    random_matrix = _sample_lhs(parameters, count1 - 1, seed)
    first_matrix = np.vstack([baseline, random_matrix])
    rows1 = _evaluate_matrix(project, first_matrix, runs1, seed, 0, workers)
    first_frame = pd.DataFrame(rows1)
    first_pareto = first_frame[_pareto_mask(first_frame)].sort_values("score")

    anchors = first_pareto.head(min(32, len(first_pareto)))
    rng = np.random.default_rng(seed + 1)
    second_values: list[list[float]] = []
    while len(second_values) < count2:
        anchor = anchors.iloc[len(second_values) % len(anchors)]
        row = []
        for item in parameters:
            span = item.maximum - item.minimum
            value = rng.normal(float(anchor[item.path]), span * 0.08)
            row.append(float(np.clip(value, item.minimum, item.maximum)))
        second_values.append(row)
    second_matrix = np.asarray(second_values)
    rows2 = _evaluate_matrix(project, second_matrix, runs2, seed + 500000, len(rows1), workers)
    trials = pd.concat([first_frame, pd.DataFrame(rows2)], ignore_index=True)
    pareto = trials[_pareto_mask(trials)].sort_values(["feasible", "score"], ascending=[False, True]).reset_index(drop=True)
    candidate_rows = [row for _, row in pareto.head(4).iterrows()]
    baseline_rows = trials[trials["candidate"] == 0]
    if not baseline_rows.empty and all(int(row["candidate"]) != 0 for row in candidate_rows):
        candidate_rows.append(baseline_rows.iloc[0])
    targets = {item.enemy_id: item for item in project.study.targets}
    checked = []
    for validation_index, row in enumerate(candidate_rows):
        values = {item.path: float(row[item.path]) for item in parameters}
        frame = _validation(project, values, validation_runs, seed + validation_index * 1000000, workers)
        is_valid, validation_score = _validation_score(frame, targets)
        checked.append((is_valid, validation_score, float(row["score"]), row, values, frame))
    checked.sort(key=lambda item: (not item[0], item[1], item[2]))
    validated, final_score, _, recommended_row, recommended_values, validation = checked[0]
    recommended = {
        "candidate": int(recommended_row["candidate"]),
        "feasible": validated,
        "score": final_score,
        "values": recommended_values,
        "note": "通过最终大样本复核的最小综合损失方案" if validated else "最终大样本复核未找到完全可行解，返回距离目标最近的 Pareto 方案",
    }
    return OptimizationResult(trials=trials, pareto=pareto, recommended=recommended, validation=validation)
