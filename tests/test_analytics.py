from __future__ import annotations

import pandas as pd

from combat_balance import build_scenario, load_project, simulate_batch
from combat_balance.analytics import wilson_interval
from combat_balance.engine import scaled_stats


def test_project_loads_and_growth_is_monotonic():
    project = load_project("configs/portfolio_arpg.yaml")
    for player in project.players.values():
        assert scaled_stats(player, 50).attack > scaled_stats(player, 1).attack
        assert scaled_stats(player, 50).hp > scaled_stats(player, 1).hp


def test_initial_draft_inherits_balanced_config():
    balanced = load_project("configs/portfolio_arpg.yaml")
    draft = load_project("configs/initial_draft.yaml")
    assert draft.players["ranger"].name == balanced.players["ranger"].name
    assert draft.players["vanguard"].stats.attack < balanced.players["vanguard"].stats.attack
    assert draft.enemies["elite"].stats.attack < balanced.enemies["elite"].stats.attack


def test_batch_is_reproducible_across_worker_counts():
    project = load_project("configs/portfolio_arpg.yaml")
    scenario = build_scenario(project, "ranger", "common", 10, "priority")
    sequential = simulate_batch(scenario, runs=20, seed=12345, workers=1)
    parallel = simulate_batch(scenario, runs=20, seed=12345, workers=2)
    pd.testing.assert_frame_equal(sequential.trials, parallel.trials)
    assert sequential.summary == parallel.summary


def test_ttk_excludes_failed_fights():
    project = load_project("configs/portfolio_arpg.yaml")
    project = project.model_copy(deep=True)
    project.formula.max_time = 0.01
    result = simulate_batch(build_scenario(project, "vanguard", "boss", 30), runs=10, seed=1)
    assert result.summary["wins"] == 0
    assert result.summary["ttk_mean"] is None
    assert result.summary["timeout_rate"] == 1


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(95, 100)
    assert low < 0.95 < high
