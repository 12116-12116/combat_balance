from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .analytics import BatchResult
from .optimizer import OptimizationResult


@dataclass
class ReportManifest:
    output_dir: Path
    report: Path
    summary_json: Path
    trials_csv: Path
    skill_stats_csv: Path
    chart_paths: list[Path]
    pareto_csv: Path | None = None


def _as_results(results: BatchResult | Iterable[BatchResult] | dict[str, BatchResult]) -> list[BatchResult]:
    if isinstance(results, BatchResult):
        return [results]
    if isinstance(results, dict):
        return list(results.values())
    return list(results)


def generate_report(
    results: BatchResult | Iterable[BatchResult] | dict[str, BatchResult],
    output_dir: str | Path,
    optimization: OptimizationResult | None = None,
    title: str = "余烬前线战斗数值分析报告",
) -> ReportManifest:
    batches = _as_results(results)
    if not batches:
        raise ValueError("报告至少需要一组批量模拟结果")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    trial_frames = []
    skill_frames = []
    for batch in batches:
        label = f"{batch.scenario.player.name}-{batch.scenario.enemy.name}-Lv{batch.scenario.level}-{batch.scenario.player_policy.name}"
        summary_rows.append({"scenario": label, **batch.summary})
        trial_frames.append(batch.trials.assign(scenario=label))
        skill_frames.append(batch.skill_stats.assign(scenario=label))
    summary_frame = pd.DataFrame(summary_rows)
    trials = pd.concat(trial_frames, ignore_index=True)
    skills = pd.concat(skill_frames, ignore_index=True)

    summary_json = destination / "summary.json"
    trials_csv = destination / "trials.csv"
    skill_csv = destination / "skill_stats.csv"
    summary_json.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    trials.to_csv(trials_csv, index=False, encoding="utf-8-sig")
    skills.to_csv(skill_csv, index=False, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    charts: list[Path] = []

    fig, axis = plt.subplots(figsize=(max(8, len(summary_frame) * 0.7), 4.8))
    axis.bar(summary_frame["scenario"], summary_frame["win_rate"], color="#2d6a4f")
    axis.set_ylim(0, 1)
    axis.set_ylabel("胜率")
    axis.set_title("场景胜率对比")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    win_chart = destination / "win_rate.png"
    fig.savefig(win_chart, dpi=160)
    plt.close(fig)
    charts.append(win_chart)

    successful = trials.dropna(subset=["ttk"])
    if not successful.empty:
        labels = list(successful["scenario"].drop_duplicates())
        groups = [successful.loc[successful["scenario"] == label, "ttk"] for label in labels]
        fig, axis = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.8))
        axis.boxplot(groups, tick_labels=labels, showfliers=False)
        axis.set_ylabel("TTK（秒）")
        axis.set_title("成功击杀场次 TTK 分布")
        axis.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        ttk_chart = destination / "ttk_distribution.png"
        fig.savefig(ttk_chart, dpi=160)
        plt.close(fig)
        charts.append(ttk_chart)

    pareto_path = None
    if optimization is not None:
        pareto_path = destination / "pareto.csv"
        optimization.pareto.to_csv(pareto_path, index=False, encoding="utf-8-sig")
        optimization.trials.to_csv(destination / "optimization_trials.csv", index=False, encoding="utf-8-sig")
        optimization.validation.to_csv(destination / "optimization_validation.csv", index=False, encoding="utf-8-sig")
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(
            optimization.trials["ttk_objective"],
            optimization.trials["win_objective"],
            alpha=0.25,
            label="候选",
        )
        axis.scatter(
            optimization.pareto["ttk_objective"],
            optimization.pareto["win_objective"],
            color="#d1495b",
            label="Pareto 前沿",
        )
        axis.set_xlabel("TTK 目标偏差")
        axis.set_ylabel("胜率目标偏差")
        axis.set_title("多目标调参结果")
        axis.legend()
        fig.tight_layout()
        pareto_chart = destination / "pareto_front.png"
        fig.savefig(pareto_chart, dpi=160)
        plt.close(fig)
        charts.append(pareto_chart)

    report_path = destination / "report.md"
    lines = [
        f"# {title}",
        "",
        "## 结论摘要",
        "",
        "本报告区分人工设定的体验目标与算法生成的候选参数。最终取值仍需由策划结合职业特色、可读性和操作反馈判断。",
        "",
        "## 场景结果",
        "",
        summary_frame[["scenario", "runs", "win_rate", "ttk_mean", "ttk_p10", "ttk_p50", "ttk_p90", "dps_mean"]].to_markdown(index=False),
        "",
        "![胜率](win_rate.png)",
    ]
    if successful.empty is False:
        lines.extend(["", "![TTK 分布](ttk_distribution.png)"])
    lines.extend([
        "",
        "## 设计判断",
        "",
        "- 胜率用于判断生存压力，TTK 用于判断战斗节奏，二者必须结合查看。",
        "- P10/P50/P90 展示随机暴击、命中和闪避带来的体验波动，平均值不能替代分布。",
        "- 技能伤害占比与 CD 利用率用于定位单技能过强、资源空转和循环断档。",
    ])
    if optimization is not None:
        lines.extend([
            "",
            "## 自动调参",
            "",
            f"推荐候选：`{optimization.recommended['candidate']}`；{optimization.recommended['note']}。",
            "",
            "![Pareto 前沿](pareto_front.png)",
            "",
            "推荐参数：",
            "",
            "```json",
            json.dumps(optimization.recommended["values"], ensure_ascii=False, indent=2),
            "```",
        ])
    lines.extend([
        "",
        "## 方法说明",
        "",
        "同一 seed 会为每场战斗派生独立随机流，因而并行数变化不会改变结果。伤害只统计目标剩余生命内的有效部分，溢出伤害单独记录。",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return ReportManifest(destination, report_path, summary_json, trials_csv, skill_csv, charts, pareto_path)
