from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import ProjectConfig, ScenarioConfig


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_project(path: str | Path) -> ProjectConfig:
    """加载并严格校验项目 YAML 配置。"""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    parent = data.pop("extends", None)
    if parent:
        parent_path = (config_path.parent / parent).resolve()
        with parent_path.open("r", encoding="utf-8") as handle:
            parent_data = yaml.safe_load(handle)
        if "extends" in parent_data:
            raise ValueError("配置继承仅支持一层")
        data = _deep_merge(parent_data, data)
    return ProjectConfig.model_validate(data)


def build_scenario(
    project: ProjectConfig,
    player_id: str | None = None,
    enemy_id: str | None = None,
    level: int = 30,
    policy_id: str | None = None,
    enemy_policy_id: str = "priority",
) -> ScenarioConfig:
    player_key = player_id or project.study.default_player
    enemy_key = enemy_id or project.study.default_enemy
    policy_key = policy_id or project.study.default_policy
    try:
        player = project.players[player_key]
        enemy = project.enemies[enemy_key]
        policy = project.policies[policy_key]
        enemy_policy = project.policies[enemy_policy_id]
    except KeyError as exc:
        raise ValueError(f"未知配置引用: {exc.args[0]}") from exc
    return ScenarioConfig(
        player=player,
        enemy=enemy,
        player_policy=policy,
        enemy_policy=enemy_policy,
        formula=project.formula,
        level=level,
    )


def get_by_path(data: dict[str, Any], path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def set_by_path(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node: Any = data
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def project_with_values(project: ProjectConfig, values: dict[str, float]) -> ProjectConfig:
    data = project.model_dump(mode="python")
    for path, value in values.items():
        set_by_path(data, path, float(value))
    return ProjectConfig.model_validate(data)
