from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .engine import BattleResult, simulate_once
from .models import ScenarioConfig


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_interval(
    values: list[float], seed: int, samples: int = 2000
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = np.empty(samples)
    for index in range(samples):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


@dataclass
class BatchResult:
    scenario: ScenarioConfig
    summary: dict[str, Any]
    trials: pd.DataFrame
    skill_stats: pd.DataFrame
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": {
                "player": self.scenario.player.id,
                "enemy": self.scenario.enemy.id,
                "level": self.scenario.level,
                "policy": self.scenario.player_policy.id,
            },
            "summary": self.summary,
            "seed": self.seed,
        }


def _seed_values(seed: int, runs: int) -> list[int]:
    sequence = np.random.SeedSequence(seed)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(runs)]


def _run_trial(scenario: ScenarioConfig, seed: int) -> BattleResult:
    return simulate_once(scenario, seed=seed, log_events=False)


def _safe_stats(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_p10": None,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": round(float(array.mean()), 6),
        f"{prefix}_std": round(float(array.std(ddof=1)) if len(array) > 1 else 0.0, 6),
        f"{prefix}_p10": round(float(np.quantile(array, 0.10)), 6),
        f"{prefix}_p50": round(float(np.quantile(array, 0.50)), 6),
        f"{prefix}_p90": round(float(np.quantile(array, 0.90)), 6),
    }


def simulate_batch(
    scenario: ScenarioConfig,
    runs: int = 1000,
    seed: int = 20260908,
    workers: int = 1,
) -> BatchResult:
    if runs <= 0:
        raise ValueError("runs 必须大于 0")
    seeds = _seed_values(seed, runs)
    if workers == 1:
        results = [_run_trial(scenario, item) for item in seeds]
    else:
        results = Parallel(n_jobs=workers, backend="loky")(
            delayed(_run_trial)(scenario, item) for item in seeds
        )

    rows = []
    for index, result in enumerate(results):
        rows.append({
            "trial": index,
            "seed": result.seed,
            "outcome": result.outcome,
            "duration": result.duration,
            "ttk": result.ttk,
            "ttd": result.ttd,
            "player_dps": result.player_dps,
            "player_damage": result.player_damage,
            "enemy_damage": result.enemy_damage,
            "player_hp": result.player_hp,
            "enemy_hp": result.enemy_hp,
            "player_overkill": result.player_overkill,
            "resource_spent": result.player_resource_spent,
            "resource_end": result.player_resource_end,
        })
    trials = pd.DataFrame(rows)
    wins = sum(result.outcome == "victory" for result in results)
    defeats = sum(result.outcome == "defeat" for result in results)
    draws = sum(result.outcome == "draw" for result in results)
    timeouts = sum(result.outcome == "timeout" for result in results)
    ttk_values = [result.duration for result in results if result.outcome == "victory"]
    ttd_values = [result.duration for result in results if result.outcome == "defeat"]
    dps_values = [result.player_dps for result in results]
    win_low, win_high = wilson_interval(wins, runs)
    ttk_low, ttk_high = bootstrap_mean_interval(ttk_values, seed + 1009)
    summary: dict[str, Any] = {
        "runs": runs,
        "wins": wins,
        "defeats": defeats,
        "draws": draws,
        "timeouts": timeouts,
        "win_rate": round(wins / runs, 6),
        "defeat_rate": round(defeats / runs, 6),
        "draw_rate": round(draws / runs, 6),
        "timeout_rate": round(timeouts / runs, 6),
        "win_rate_ci95_low": round(win_low, 6),
        "win_rate_ci95_high": round(win_high, 6),
        "ttk_mean_ci95_low": round(ttk_low, 6) if ttk_low is not None else None,
        "ttk_mean_ci95_high": round(ttk_high, 6) if ttk_high is not None else None,
        "damage_taken_mean": round(float(trials["enemy_damage"].mean()), 6),
        "resource_spent_mean": round(float(trials["resource_spent"].mean()), 6),
        "resource_end_mean": round(float(trials["resource_end"].mean()), 6),
    }
    summary.update(_safe_stats(ttk_values, "ttk"))
    summary.update(_safe_stats(ttd_values, "ttd"))
    summary.update(_safe_stats(dps_values, "dps"))

    skill_rows: list[dict[str, Any]] = []
    skill_ids = {skill.id for skill in scenario.player.skills + [scenario.player.basic_attack]}
    total_player_damage = sum(result.player_damage for result in results)
    total_duration = sum(result.duration for result in results)
    for skill_id in sorted(skill_ids):
        entries = [result.skill_metrics["player"][skill_id] for result in results]
        casts = sum(entry["casts"] for entry in entries)
        hits = sum(entry["hits"] for entry in entries)
        crits = sum(entry["crits"] for entry in entries)
        damage = sum(entry["damage"] for entry in entries)
        cooldown = entries[0]["cooldown"] if entries else 0.0
        theoretical = runs if cooldown <= 0 else sum(math.floor(result.duration / cooldown) + 1 for result in results)
        skill_rows.append({
            "skill_id": skill_id,
            "casts": int(casts),
            "casts_per_battle": casts / runs,
            "hit_rate": hits / casts if casts else 0.0,
            "crit_rate_on_hit": crits / hits if hits else 0.0,
            "effective_damage": damage,
            "damage_share": damage / total_player_damage if total_player_damage else 0.0,
            "cooldown_utilization": min(1.0, casts / theoretical) if theoretical else 0.0,
            "effective_dps": damage / total_duration if total_duration else 0.0,
        })
    skill_stats = pd.DataFrame(skill_rows)
    return BatchResult(scenario=scenario, summary=summary, trials=trials, skill_stats=skill_stats, seed=seed)


def batch_summary_json(result: BatchResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
