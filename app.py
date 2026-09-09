from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from combat_balance import (
    analyze_sensitivity,
    build_scenario,
    generate_report,
    load_project,
    optimize_balance,
    simulate_batch,
    simulate_once,
)
from combat_balance.engine import scaled_stats
from combat_balance.config import project_with_values


st.set_page_config(page_title="余烬前线·战斗数值实验室", page_icon="⚔️", layout="wide")
st.title("⚔️ 余烬前线·战斗数值实验室")
st.caption("实时制 PvE 模拟 · 蒙特卡洛分析 · 成长曲线 · 自动调参")


@st.cache_resource
def get_project(path: str):
    return load_project(path)


config_path = st.sidebar.text_input("配置文件", "configs/portfolio_arpg.yaml")
try:
    project = get_project(config_path)
except Exception as exc:
    st.error(f"配置加载失败：{exc}")
    st.stop()

player_id = st.sidebar.selectbox("职业", list(project.players), format_func=lambda key: project.players[key].name)
enemy_id = st.sidebar.selectbox("敌人", list(project.enemies), index=1, format_func=lambda key: project.enemies[key].name)
level = st.sidebar.select_slider("等级", options=list(range(1, 51)), value=30)
policy_id = st.sidebar.selectbox("战斗策略", list(project.policies), format_func=lambda key: project.policies[key].name)
runs = st.sidebar.number_input("批量模拟次数", min_value=50, max_value=10000, value=500, step=50)
seed = st.sidebar.number_input("随机种子", min_value=0, value=20260908, step=1)
st.sidebar.divider()
st.sidebar.caption("当前场景临时参数")
player_attack_scale = st.sidebar.slider("玩家攻击倍率", 0.7, 1.3, 1.0, 0.01)
enemy_hp_scale = st.sidebar.slider("敌人生命倍率", 0.7, 1.3, 1.0, 0.01)
enemy_attack_scale = st.sidebar.slider("敌人攻击倍率", 0.7, 1.3, 1.0, 0.01)


def scenario_for(selected_player: str, selected_enemy: str, selected_level: int, selected_policy: str):
    values = {
        f"players.{selected_player}.stats.attack": project.players[selected_player].stats.attack * player_attack_scale,
        f"enemies.{selected_enemy}.stats.hp": project.enemies[selected_enemy].stats.hp * enemy_hp_scale,
        f"enemies.{selected_enemy}.stats.attack": project.enemies[selected_enemy].stats.attack * enemy_attack_scale,
    }
    edited_project = project_with_values(project, values)
    return build_scenario(edited_project, selected_player, selected_enemy, selected_level, selected_policy)


scenario = scenario_for(player_id, enemy_id, level, policy_id)

overview_tab, single_tab, batch_tab, growth_tab, sensitivity_tab, optimize_tab = st.tabs(
    ["项目总览", "单场时间轴", "批量对比", "成长曲线", "敏感性分析", "自动调参与报告"]
)

with overview_tab:
    st.subheader("设计目标")
    target_rows = [
        {
            "敌人": project.enemies[target.enemy_id].name,
            "目标 TTK": f"{target.ttk_min:.0f}～{target.ttk_max:.0f} 秒",
            "阵容平均胜率": f"{target.win_rate_min:.0%}～{target.win_rate_max:.0%}",
            "体验定位": project.enemies[target.enemy_id].archetype,
        }
        for target in project.study.targets
    ]
    target_frame = pd.DataFrame(target_rows)
    st.dataframe(target_frame, width="stretch", hide_index=True)
    cols = st.columns(3)
    for column, character in zip(cols, project.players.values(), strict=True):
        stats = scaled_stats(character, level)
        with column:
            st.markdown(f"### {character.name}")
            st.caption(character.archetype)
            st.metric("攻击", f"{stats.attack:.0f}")
            st.metric("生命", f"{stats.hp:.0f}")
            st.metric("防御", f"{stats.defense:.0f}")
    st.info("体验目标由策划定义；模拟与优化只用于验证和提供候选，不替代最终设计判断。")

with single_tab:
    if st.button("运行单场战斗", type="primary", key="single"):
        result = simulate_once(scenario, int(seed), log_events=True)
        st.session_state["single_result"] = result
    result = st.session_state.get("single_result")
    if result:
        columns = st.columns(5)
        outcome_names = {"victory": "胜利", "defeat": "失败", "draw": "平局", "timeout": "超时"}
        columns[0].metric("结果", outcome_names[result.outcome])
        columns[1].metric("时长", f"{result.duration:.2f}s")
        columns[2].metric("玩家 DPS", f"{result.player_dps:.1f}")
        columns[3].metric("玩家剩余 HP", f"{result.player_hp:.0f}")
        columns[4].metric("敌人剩余 HP", f"{result.enemy_hp:.0f}")
        events = pd.DataFrame(result.events)
        if not events.empty:
            damage_events = events[events["damage"] > 0]
            figure = px.scatter(
                damage_events,
                x="time",
                y="damage",
                color="actor",
                size="damage",
                hover_data=["skill", "result", "target_hp"],
                title="伤害事件时间轴",
            )
            st.plotly_chart(figure, width="stretch")
            st.dataframe(events, width="stretch", hide_index=True)

with batch_tab:
    if st.button("比较三个职业", type="primary", key="batch"):
        with st.spinner("正在运行独立随机流模拟……"):
            batches = {}
            for index, (key, character) in enumerate(project.players.items()):
                item = scenario_for(key, enemy_id, level, character.default_policy)
                batches[key] = simulate_batch(item, int(runs), int(seed) + index, workers=1)
            st.session_state["batches"] = batches
    batches = st.session_state.get("batches")
    if batches:
        comparison = pd.DataFrame([
            {
                "职业": batch.scenario.player.name,
                "胜率": batch.summary["win_rate"],
                "平均TTK": batch.summary["ttk_mean"],
                "P10": batch.summary["ttk_p10"],
                "P50": batch.summary["ttk_p50"],
                "P90": batch.summary["ttk_p90"],
                "平均DPS": batch.summary["dps_mean"],
            }
            for batch in batches.values()
        ])
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.plotly_chart(px.bar(comparison, x="职业", y="胜率", range_y=[0, 1], title="职业胜率"), width="stretch")
        all_trials = pd.concat([
            batch.trials.assign(职业=batch.scenario.player.name) for batch in batches.values()
        ])
        st.plotly_chart(px.box(all_trials.dropna(subset=["ttk"]), x="职业", y="ttk", title="TTK 分布"), width="stretch")

with growth_tab:
    rows = []
    for checkpoint in range(1, 51):
        for character in project.players.values():
            stats = scaled_stats(character, checkpoint)
            rows.append({"等级": checkpoint, "职业": character.name, "攻击": stats.attack, "防御": stats.defense, "生命": stats.hp})
    growth = pd.DataFrame(rows)
    metric = st.radio("成长指标", ["攻击", "生命", "防御"], horizontal=True)
    st.plotly_chart(px.line(growth, x="等级", y=metric, color="职业", title=f"1～50 级{metric}成长曲线"), width="stretch")
    st.dataframe(growth[growth["等级"].isin(project.study.level_checkpoints)], width="stretch", hide_index=True)

with sensitivity_tab:
    default_paths = [item.path for item in project.optimization.parameters[:4]]
    selected_paths = st.multiselect(
        "分析参数",
        [item.path for item in project.optimization.parameters],
        default=default_paths,
    )
    if st.button("运行敏感性分析", type="primary", key="sensitivity", disabled=not selected_paths):
        with st.spinner("正在计算局部弹性与全局相关性……"):
            sensitivity = analyze_sensitivity(
                project,
                selected_paths,
                runs=min(int(runs), 300),
                samples=32,
                seed=int(seed),
            )
            st.session_state["sensitivity"] = sensitivity
    sensitivity = st.session_state.get("sensitivity")
    if sensitivity:
        st.dataframe(sensitivity.oat, width="stretch", hide_index=True)
        heatmap = sensitivity.correlations.pivot(index="parameter", columns="metric", values="spearman")
        st.plotly_chart(px.imshow(heatmap, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Spearman 参数相关性"), width="stretch")

with optimize_tab:
    mode = st.radio("搜索规模", ["快速验证", "完整研究"], horizontal=True)
    if st.button("运行多目标调参", type="primary", key="optimize"):
        overrides = {}
        if mode == "快速验证":
            overrides = dict(stage1_candidates=12, stage1_runs=45, stage2_candidates=8, stage2_runs=90, final_runs=180)
        with st.spinner("正在搜索 Pareto 候选……完整研究可能需要数分钟。"):
            optimization = optimize_balance(project, seed=int(seed), workers=1, **overrides)
            st.session_state["optimization"] = optimization
    optimization = st.session_state.get("optimization")
    if optimization:
        st.success(optimization.recommended["note"])
        st.json(optimization.recommended)
        st.plotly_chart(
            px.scatter(
                optimization.trials,
                x="ttk_objective",
                y="win_objective",
                color="class_spread",
                hover_data=["candidate", "score", "feasible"],
                title="候选方案目标空间",
            ),
            width="stretch",
        )
        st.dataframe(optimization.validation, width="stretch", hide_index=True)

    st.divider()
    st.subheader("导出当前对比报告")
    if st.button("生成 Markdown / CSV / PNG 报告包", key="report"):
        current_batches = st.session_state.get("batches")
        if not current_batches:
            current_batches = {}
            for index, (key, character) in enumerate(project.players.items()):
                item = scenario_for(key, enemy_id, level, character.default_policy)
                current_batches[key] = simulate_batch(item, min(int(runs), 1000), int(seed) + index)
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = generate_report(
                current_batches,
                temp_dir,
                st.session_state.get("optimization"),
            )
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in Path(temp_dir).iterdir():
                    archive.write(path, arcname=path.name)
            st.download_button(
                "下载分析报告 ZIP",
                data=buffer.getvalue(),
                file_name="combat_balance_report.zip",
                mime="application/zip",
            )
