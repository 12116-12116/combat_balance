from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analytics import simulate_batch
from .config import build_scenario, load_project
from .engine import simulate_once
from .optimizer import optimize_balance
from .reporting import generate_report
from .sensitivity import analyze_sensitivity


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实时制 PvE 战斗数值模拟与分析工具")
    parser.add_argument("--config", default="configs/portfolio_arpg.yaml", help="项目 YAML 配置")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="运行单场或批量模拟")
    simulate.add_argument("--player")
    simulate.add_argument("--enemy")
    simulate.add_argument("--level", type=int, default=30)
    simulate.add_argument("--policy")
    simulate.add_argument("--runs", type=int, default=1000)
    simulate.add_argument("--seed", type=int, default=20260908)
    simulate.add_argument("--workers", type=int, default=1)
    simulate.add_argument("--single", action="store_true")
    simulate.add_argument("--output", type=Path)

    analyze = subparsers.add_parser("analyze", help="运行敏感性分析")
    analyze.add_argument("--runs", type=int, default=300)
    analyze.add_argument("--samples", type=int, default=64)
    analyze.add_argument("--seed", type=int, default=20260908)
    analyze.add_argument("--workers", type=int, default=1)
    analyze.add_argument("--parameters", nargs="*")
    analyze.add_argument("--output", type=Path, default=Path("artifacts/sensitivity"))

    optimize = subparsers.add_parser("optimize", help="运行多目标自动调参")
    optimize.add_argument("--seed", type=int, default=20260908)
    optimize.add_argument("--workers", type=int, default=1)
    optimize.add_argument("--quick", action="store_true", help="用于流程验证的小规模搜索")
    optimize.add_argument("--output", type=Path, default=Path("artifacts/optimization"))

    report = subparsers.add_parser("report", help="生成成长与战斗分析报告")
    report.add_argument("--runs", type=int)
    report.add_argument("--seed", type=int, default=20260908)
    report.add_argument("--workers", type=int, default=1)
    report.add_argument("--output", type=Path, default=Path("artifacts/report"))
    return parser


def _scenario(project, args):
    return build_scenario(
        project,
        args.player or project.study.default_player,
        args.enemy or project.study.default_enemy,
        args.level,
        args.policy or project.study.default_policy,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _base_parser()
    args = parser.parse_args(argv)
    project = load_project(args.config)

    if args.command == "simulate":
        scenario = _scenario(project, args)
        if args.single:
            data = simulate_once(scenario, args.seed).to_dict()
        else:
            batch = simulate_batch(scenario, args.runs, args.seed, args.workers)
            data = batch.to_dict()
            if args.output:
                generate_report(batch, args.output)
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.command == "analyze":
        result = analyze_sensitivity(
            project,
            args.parameters,
            runs=args.runs,
            samples=args.samples,
            seed=args.seed,
            workers=args.workers,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        result.oat.to_csv(args.output / "oat.csv", index=False, encoding="utf-8-sig")
        result.correlations.to_csv(args.output / "correlations.csv", index=False, encoding="utf-8-sig")
        result.samples.to_csv(args.output / "samples.csv", index=False, encoding="utf-8-sig")
        print(result.correlations.to_string(index=False))
        return 0

    if args.command == "optimize":
        overrides = {}
        if args.quick:
            overrides = {
                "stage1_candidates": 12,
                "stage1_runs": 45,
                "stage2_candidates": 8,
                "stage2_runs": 90,
                "final_runs": 180,
            }
        result = optimize_balance(project, seed=args.seed, workers=args.workers, **overrides)
        args.output.mkdir(parents=True, exist_ok=True)
        result.trials.to_csv(args.output / "optimization_trials.csv", index=False, encoding="utf-8-sig")
        result.pareto.to_csv(args.output / "pareto.csv", index=False, encoding="utf-8-sig")
        result.validation.to_csv(args.output / "validation.csv", index=False, encoding="utf-8-sig")
        (args.output / "recommended.json").write_text(
            json.dumps(result.recommended, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result.recommended, ensure_ascii=False, indent=2))
        return 0

    if args.command == "report":
        runs = args.runs or project.study.report_runs
        batches = []
        scenario_index = 0
        for level in project.study.level_checkpoints:
            for enemy_id in project.enemies:
                for player_id, player in project.players.items():
                    scenario = build_scenario(project, player_id, enemy_id, level, player.default_policy)
                    batches.append(
                        simulate_batch(
                            scenario,
                            runs=runs,
                            seed=args.seed + scenario_index,
                            workers=args.workers,
                        )
                    )
                    scenario_index += 1
        manifest = generate_report(batches, args.output)
        print(str(manifest.report.resolve()))
        return 0

    parser.error("未知命令")
    return 2

