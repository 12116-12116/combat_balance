from __future__ import annotations

from pathlib import Path

from combat_balance import (
    analyze_sensitivity,
    build_scenario,
    generate_report,
    load_project,
    optimize_balance,
    simulate_batch,
)
from combat_balance.cli import main


def test_sensitivity_produces_local_and_global_results():
    project = load_project("configs/portfolio_arpg.yaml")
    path = project.optimization.parameters[0].path
    result = analyze_sensitivity(project, [path], runs=10, samples=4, seed=7)
    assert len(result.oat) == 4
    assert set(result.correlations["metric"]) == {"ttk_mean", "win_rate", "dps_mean"}


def test_quick_optimizer_and_report(tmp_path: Path):
    project = load_project("configs/portfolio_arpg.yaml")
    result = optimize_balance(
        project,
        workers=1,
        stage1_candidates=3,
        stage1_runs=18,
        stage2_candidates=2,
        stage2_runs=18,
        final_runs=18,
        seed=8,
    )
    assert len(result.trials) == 5
    assert not result.pareto.empty
    assert set(result.recommended["values"]) == {item.path for item in project.optimization.parameters}

    batch = simulate_batch(build_scenario(project, "vanguard", "common", 1), runs=10, seed=9)
    manifest = generate_report(batch, tmp_path, optimization=result)
    assert manifest.report.exists()
    assert manifest.summary_json.exists()
    assert manifest.trials_csv.exists()
    assert manifest.pareto_csv and manifest.pareto_csv.exists()


def test_cli_simulate_entrypoint(capsys):
    exit_code = main([
        "--config", "configs/portfolio_arpg.yaml",
        "simulate", "--runs", "3", "--seed", "10",
    ])
    assert exit_code == 0
    assert '"runs": 3' in capsys.readouterr().out


def test_streamlit_dashboard_smoke():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 6
    app.button[0].click().run()
    assert not app.exception
    assert any(metric.label == "结果" for metric in app.metric)
