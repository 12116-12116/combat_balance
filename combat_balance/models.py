from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StatBlock(StrictModel):
    attack: float = Field(gt=0)
    defense: float = Field(ge=0)
    hp: float = Field(gt=0)
    accuracy: float = Field(default=1.0, ge=0, le=2)
    dodge: float = Field(default=0.0, ge=0, le=1)
    crit_rate: float = Field(default=0.0, ge=0, le=1)
    crit_damage: float = Field(default=0.5, ge=0)
    resource_max: float = Field(default=0.0, ge=0)
    resource_regen: float = Field(default=0.0, ge=0)


class GrowthRates(StrictModel):
    attack: float = Field(default=0.0, ge=-0.99, le=1)
    defense: float = Field(default=0.0, ge=-0.99, le=1)
    hp: float = Field(default=0.0, ge=-0.99, le=1)
    resource_max: float = Field(default=0.0, ge=-0.99, le=1)
    resource_regen: float = Field(default=0.0, ge=-0.99, le=1)


class BuffConfig(StrictModel):
    id: str
    name: str
    duration: float = Field(gt=0)
    modifiers: dict[str, float] = Field(default_factory=dict)
    stacking: Literal["additive", "multiplicative", "replace"] = "additive"


class SkillConfig(StrictModel):
    id: str
    name: str
    multiplier: float = Field(default=0.0, ge=0)
    cooldown: float = Field(default=0.0, ge=0)
    cast_time: float = Field(default=0.1, ge=0)
    recovery: float = Field(default=0.5, ge=0)
    hit_rate: float = Field(default=1.0, ge=0, le=1)
    crit_bonus: float = Field(default=0.0, ge=-1, le=1)
    resource_cost: float = Field(default=0.0, ge=0)
    priority: int = 100
    tags: list[str] = Field(default_factory=list)
    apply_buff: BuffConfig | None = None

    @model_validator(mode="after")
    def action_advances_time(self) -> "SkillConfig":
        if self.cast_time + self.recovery <= 0:
            raise ValueError("技能的 cast_time 与 recovery 不能同时为 0")
        return self


class CharacterTemplate(StrictModel):
    id: str
    name: str
    archetype: str
    stats: StatBlock
    growth: GrowthRates = Field(default_factory=GrowthRates)
    basic_attack: SkillConfig
    skills: list[SkillConfig] = Field(default_factory=list)
    default_policy: str = "priority"

    @model_validator(mode="after")
    def unique_skills(self) -> "CharacterTemplate":
        ids = [self.basic_attack.id, *(skill.id for skill in self.skills)]
        if len(ids) != len(set(ids)):
            raise ValueError(f"角色 {self.id} 存在重复技能 id")
        return self


class PolicyConfig(StrictModel):
    id: str
    name: str
    kind: Literal["priority", "burst", "conservative"]
    priority: list[str] = Field(default_factory=list)
    burst_tags: list[str] = Field(default_factory=lambda: ["burst"])
    hold_window: float = Field(default=1.5, ge=0, le=10)
    hp_threshold: float = Field(default=0.35, ge=0, le=1)
    resource_floor: float = Field(default=0.2, ge=0, le=1)


class FormulaConfig(StrictModel):
    defense_k_base: float = Field(default=100, gt=0)
    defense_k_per_level: float = Field(default=10, ge=0)
    max_time: float = Field(default=150, gt=0)
    max_events: int = Field(default=10000, gt=0)


class TargetRange(StrictModel):
    enemy_id: str
    ttk_min: float = Field(gt=0)
    ttk_max: float = Field(gt=0)
    win_rate_min: float = Field(ge=0, le=1)
    win_rate_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_ranges(self) -> "TargetRange":
        if self.ttk_min >= self.ttk_max or self.win_rate_min > self.win_rate_max:
            raise ValueError("目标区间上下界错误")
        return self


class OptimizationParameter(StrictModel):
    path: str
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def valid_bounds(self) -> "OptimizationParameter":
        if self.minimum >= self.maximum:
            raise ValueError(f"优化参数 {self.path} 的上下界错误")
        return self


class OptimizationConfig(StrictModel):
    parameters: list[OptimizationParameter] = Field(default_factory=list)
    levels: list[int] = Field(default_factory=lambda: [30])
    stage1_candidates: int = Field(default=512, gt=1)
    stage1_runs: int = Field(default=300, gt=0)
    stage2_candidates: int = Field(default=256, gt=1)
    stage2_runs: int = Field(default=1000, gt=0)
    final_runs: int = Field(default=10000, gt=0)


class StudyConfig(StrictModel):
    level_checkpoints: list[int] = Field(default_factory=lambda: [1, 10, 20, 30, 40, 50])
    report_runs: int = Field(default=1000, gt=0)
    default_player: str
    default_enemy: str
    default_policy: str = "priority"
    targets: list[TargetRange]


class ProjectConfig(StrictModel):
    name: str
    version: str = "1.0"
    formula: FormulaConfig = Field(default_factory=FormulaConfig)
    policies: dict[str, PolicyConfig]
    players: dict[str, CharacterTemplate]
    enemies: dict[str, CharacterTemplate]
    study: StudyConfig
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    @model_validator(mode="after")
    def references_exist(self) -> "ProjectConfig":
        for key, value in self.players.items():
            if key != value.id:
                raise ValueError(f"players 键 {key} 与 id {value.id} 不一致")
        for key, value in self.enemies.items():
            if key != value.id:
                raise ValueError(f"enemies 键 {key} 与 id {value.id} 不一致")
        if self.study.default_player not in self.players:
            raise ValueError("默认玩家不存在")
        if self.study.default_enemy not in self.enemies:
            raise ValueError("默认敌人不存在")
        if self.study.default_policy not in self.policies:
            raise ValueError("默认策略不存在")
        missing = {target.enemy_id for target in self.study.targets} - set(self.enemies)
        if missing:
            raise ValueError(f"目标引用了未知敌人: {sorted(missing)}")
        return self


class ScenarioConfig(StrictModel):
    player: CharacterTemplate
    enemy: CharacterTemplate
    player_policy: PolicyConfig
    enemy_policy: PolicyConfig
    formula: FormulaConfig
    level: int = Field(ge=1, le=1000)

