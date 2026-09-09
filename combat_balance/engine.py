from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .models import BuffConfig, CharacterTemplate, PolicyConfig, ScenarioConfig, SkillConfig, StatBlock


@dataclass
class ActiveBuff:
    config: BuffConfig
    expires_at: float
    applied_at: float


@dataclass
class ActorState:
    side: Literal["player", "enemy"]
    template: CharacterTemplate
    level: int
    stats: StatBlock
    hp: float
    resource: float
    last_resource_time: float = 0.0
    cooldowns: dict[str, float] = field(default_factory=dict)
    buffs: list[ActiveBuff] = field(default_factory=list)
    skill_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    overkill: float = 0.0
    resource_spent: float = 0.0

    @property
    def alive(self) -> bool:
        return self.hp > 1e-9

    @property
    def hp_ratio(self) -> float:
        return max(0.0, self.hp) / self.stats.hp

    @property
    def skills(self) -> list[SkillConfig]:
        return [*self.template.skills, self.template.basic_attack]

    def update_resource(self, current_time: float) -> None:
        elapsed = max(0.0, current_time - self.last_resource_time)
        self.resource = min(
            self.stats.resource_max,
            self.resource + elapsed * self.stats.resource_regen,
        )
        self.last_resource_time = current_time

    def purge_buffs(self, current_time: float) -> None:
        self.buffs = [buff for buff in self.buffs if current_time < buff.expires_at - 1e-12]


@dataclass
class BattleResult:
    outcome: Literal["victory", "defeat", "draw", "timeout"]
    winner: str | None
    duration: float
    player_hp: float
    enemy_hp: float
    player_damage: float
    enemy_damage: float
    player_overkill: float
    enemy_overkill: float
    player_resource_end: float
    enemy_resource_end: float
    player_resource_spent: float
    enemy_resource_spent: float
    events: list[dict[str, Any]]
    skill_metrics: dict[str, dict[str, dict[str, float]]]
    seed: int

    @property
    def ttk(self) -> float | None:
        return self.duration if self.outcome == "victory" else None

    @property
    def ttd(self) -> float | None:
        return self.duration if self.outcome == "defeat" else None

    @property
    def player_dps(self) -> float:
        return self.player_damage / self.duration if self.duration > 0 else 0.0

    def to_dict(self, include_events: bool = True) -> dict[str, Any]:
        data = {
            "outcome": self.outcome,
            "winner": self.winner,
            "duration": self.duration,
            "ttk": self.ttk,
            "ttd": self.ttd,
            "player_hp": self.player_hp,
            "enemy_hp": self.enemy_hp,
            "player_damage": self.player_damage,
            "enemy_damage": self.enemy_damage,
            "player_overkill": self.player_overkill,
            "enemy_overkill": self.enemy_overkill,
            "player_resource_end": self.player_resource_end,
            "enemy_resource_end": self.enemy_resource_end,
            "player_resource_spent": self.player_resource_spent,
            "enemy_resource_spent": self.enemy_resource_spent,
            "player_dps": self.player_dps,
            "skill_metrics": self.skill_metrics,
            "seed": self.seed,
        }
        if include_events:
            data["events"] = self.events
        return data


def scaled_stats(template: CharacterTemplate, level: int) -> StatBlock:
    base = template.stats
    growth = template.growth
    steps = level - 1
    return StatBlock(
        attack=base.attack * (1 + growth.attack) ** steps,
        defense=base.defense * (1 + growth.defense) ** steps,
        hp=base.hp * (1 + growth.hp) ** steps,
        accuracy=base.accuracy,
        dodge=base.dodge,
        crit_rate=base.crit_rate,
        crit_damage=base.crit_damage,
        resource_max=base.resource_max * (1 + growth.resource_max) ** steps,
        resource_regen=base.resource_regen * (1 + growth.resource_regen) ** steps,
    )


def _actor(side: Literal["player", "enemy"], template: CharacterTemplate, level: int) -> ActorState:
    stats = scaled_stats(template, level)
    metrics = {
        skill.id: {
            "casts": 0.0,
            "hits": 0.0,
            "crits": 0.0,
            "damage": 0.0,
            "cooldown": skill.cooldown,
        }
        for skill in [*template.skills, template.basic_attack]
    }
    return ActorState(
        side=side,
        template=template,
        level=level,
        stats=stats,
        hp=stats.hp,
        resource=stats.resource_max,
        skill_metrics=metrics,
    )


def _modifier(actor: ActorState, key: str, current_time: float) -> float:
    actor.purge_buffs(current_time)
    additive = 0.0
    multiplier = 1.0
    replacement: float | None = None
    for active in actor.buffs:
        if key not in active.config.modifiers:
            continue
        value = active.config.modifiers[key]
        if active.config.stacking == "additive":
            additive += value
        elif active.config.stacking == "multiplicative":
            multiplier *= 1 + value
        else:
            replacement = value
    if replacement is not None:
        return replacement
    return (1 + additive) * multiplier


def _flat_modifier(actor: ActorState, key: str, current_time: float) -> float:
    actor.purge_buffs(current_time)
    values = [
        active.config.modifiers[key]
        for active in actor.buffs
        if key in active.config.modifiers
    ]
    return sum(values)


def _ordered_skills(actor: ActorState, policy: PolicyConfig) -> list[SkillConfig]:
    by_id = {skill.id: skill for skill in actor.skills}
    ordered: list[SkillConfig] = []
    for skill_id in policy.priority:
        skill = by_id.pop(skill_id, None)
        if skill is not None:
            ordered.append(skill)
    ordered.extend(sorted(by_id.values(), key=lambda item: (item.priority, item.id)))
    return ordered


def _choose_action(
    actor: ActorState,
    policy: PolicyConfig,
    current_time: float,
) -> tuple[SkillConfig | None, float | None]:
    actor.update_resource(current_time)
    ordered = _ordered_skills(actor, policy)

    def usable(skill: SkillConfig) -> bool:
        return actor.cooldowns.get(skill.id, 0.0) <= current_time + 1e-12 and skill.resource_cost <= actor.resource + 1e-12

    available = [skill for skill in ordered if usable(skill)]
    if policy.kind == "conservative":
        if actor.hp_ratio <= policy.hp_threshold:
            defensive = [skill for skill in available if "defensive" in skill.tags or "heal" in skill.tags]
            if defensive:
                return defensive[0], None
        else:
            non_defensive = [skill for skill in available if "defensive" not in skill.tags and "heal" not in skill.tags]
            if non_defensive:
                available = non_defensive
        floor = actor.stats.resource_max * policy.resource_floor
        sustainable = [
            skill for skill in available
            if skill.id == actor.template.basic_attack.id
            or skill.resource_cost == 0
            or actor.resource - skill.resource_cost >= floor
            or "defensive" in skill.tags
        ]
        if sustainable:
            return sustainable[0], None

    if policy.kind == "burst":
        buff_skills = [skill for skill in available if skill.apply_buff and "burst_buff" in skill.tags]
        if buff_skills:
            return buff_skills[0], None
        active_damage_buff = _modifier(actor, "attack_pct", current_time) > 1.0 + 1e-12
        burst_ready = [skill for skill in available if set(skill.tags) & set(policy.burst_tags)]
        if active_damage_buff and burst_ready:
            return burst_ready[0], None
        future_buffs = [
            actor.cooldowns.get(skill.id, 0.0)
            for skill in ordered
            if skill.apply_buff and "burst_buff" in skill.tags
            and 0 < actor.cooldowns.get(skill.id, 0.0) - current_time <= policy.hold_window
        ]
        if future_buffs and burst_ready:
            return None, min(future_buffs)

    if available:
        return available[0], None

    future = [
        ready_at for skill_id, ready_at in actor.cooldowns.items()
        if ready_at > current_time and next((s for s in actor.skills if s.id == skill_id), None) is not None
    ]
    return None, min(future) if future else current_time + 0.05


def _calculate_impact(
    source: ActorState,
    target: ActorState,
    skill: SkillConfig,
    current_time: float,
    rng: np.random.Generator,
    defense_k_base: float,
    defense_k_per_level: float,
) -> dict[str, Any]:
    source.skill_metrics[skill.id]["casts"] += 1
    if skill.multiplier == 0 and skill.apply_buff is not None:
        source.skill_metrics[skill.id]["hits"] += 1
        return {"result": "buff", "raw_damage": 0.0, "damage": 0.0, "overkill": 0.0, "critical": False}
    hit_chance = min(1.0, max(0.0, skill.hit_rate * source.stats.accuracy + _flat_modifier(source, "accuracy_flat", current_time)))
    if rng.random() > hit_chance:
        return {"result": "miss", "raw_damage": 0.0, "damage": 0.0, "overkill": 0.0, "critical": False}

    dodge = min(1.0, max(0.0, target.stats.dodge + _flat_modifier(target, "dodge_flat", current_time)))
    if rng.random() < dodge:
        return {"result": "dodged", "raw_damage": 0.0, "damage": 0.0, "overkill": 0.0, "critical": False}

    source.skill_metrics[skill.id]["hits"] += 1
    attack = source.stats.attack * _modifier(source, "attack_pct", current_time)
    defense = target.stats.defense * _modifier(target, "defense_pct", current_time)
    k_value = defense_k_base + source.level * defense_k_per_level
    reduction_multiplier = k_value / (k_value + max(0.0, defense))
    raw = attack * skill.multiplier * reduction_multiplier

    crit_rate = min(1.0, max(0.0, source.stats.crit_rate + skill.crit_bonus + _flat_modifier(source, "crit_rate_flat", current_time)))
    critical = rng.random() < crit_rate
    if critical:
        raw *= 1 + max(0.0, source.stats.crit_damage + _flat_modifier(source, "crit_damage_flat", current_time))
        source.skill_metrics[skill.id]["crits"] += 1

    raw *= _modifier(source, "damage_done_pct", current_time)
    raw *= _modifier(target, "damage_taken_pct", current_time)
    effective = min(max(0.0, target.hp), raw)
    overkill = max(0.0, raw - effective)
    source.skill_metrics[skill.id]["damage"] += effective
    return {
        "result": "critical" if critical else "normal",
        "raw_damage": raw,
        "damage": effective,
        "overkill": overkill,
        "critical": critical,
    }


def simulate_once(scenario: ScenarioConfig, seed: int = 0, log_events: bool = True) -> BattleResult:
    """运行一场可复现的双边实时战斗。"""
    rng = np.random.default_rng(seed)
    player = _actor("player", scenario.player, scenario.level)
    enemy = _actor("enemy", scenario.enemy, scenario.level)
    actors = {"player": player, "enemy": enemy}
    policies = {"player": scenario.player_policy, "enemy": scenario.enemy_policy}
    event_heap: list[tuple[float, int, str, str, str | None]] = []
    sequence = itertools.count()
    heapq.heappush(event_heap, (0.0, next(sequence), "ready", "player", None))
    heapq.heappush(event_heap, (0.0, next(sequence), "ready", "enemy", None))
    event_log: list[dict[str, Any]] = []
    processed = 0
    end_time = scenario.formula.max_time
    outcome: Literal["victory", "defeat", "draw", "timeout"] = "timeout"

    while event_heap and processed < scenario.formula.max_events:
        current_time = event_heap[0][0]
        if current_time > scenario.formula.max_time + 1e-12:
            break
        end_time = current_time
        group: list[tuple[float, int, str, str, str | None]] = []
        while event_heap and math.isclose(event_heap[0][0], current_time, abs_tol=1e-12):
            group.append(heapq.heappop(event_heap))
        processed += len(group)

        for actor in actors.values():
            actor.update_resource(current_time)
            actor.purge_buffs(current_time)

        impacts = [event for event in group if event[2] == "impact"]
        readies = [event for event in group if event[2] == "ready"]

        pending_damage: dict[str, float] = {"player": 0.0, "enemy": 0.0}
        impact_rows: list[tuple[ActorState, ActorState, SkillConfig, dict[str, Any]]] = []
        for _, _, _, side, skill_id in impacts:
            source = actors[side]
            target = actors["enemy" if side == "player" else "player"]
            if not source.alive or skill_id is None:
                continue
            skill = next(skill for skill in source.skills if skill.id == skill_id)
            result = _calculate_impact(
                source,
                target,
                skill,
                current_time,
                rng,
                scenario.formula.defense_k_base,
                scenario.formula.defense_k_per_level,
            )
            pending_damage[target.side] += result["damage"]
            impact_rows.append((source, target, skill, result))

        for side, damage in pending_damage.items():
            actors[side].hp = max(0.0, actors[side].hp - damage)
            actors[side].damage_taken += damage

        for source, target, skill, result in impact_rows:
            source.damage_dealt += result["damage"]
            source.overkill += result["overkill"]
            if skill.apply_buff is not None:
                source.buffs.append(
                    ActiveBuff(skill.apply_buff, current_time + skill.apply_buff.duration, current_time)
                )
            if log_events:
                event_log.append({
                    "time": round(current_time, 6),
                    "actor": source.side,
                    "target": target.side,
                    "skill_id": skill.id,
                    "skill": skill.name,
                    "result": result["result"],
                    "raw_damage": round(result["raw_damage"], 6),
                    "damage": round(result["damage"], 6),
                    "overkill": round(result["overkill"], 6),
                    "target_hp": round(target.hp, 6),
                    "resource": round(source.resource, 6),
                })

        if not player.alive or not enemy.alive:
            if not player.alive and not enemy.alive:
                outcome = "draw"
            elif enemy.alive:
                outcome = "defeat"
            else:
                outcome = "victory"
            break

        for _, _, _, side, _ in readies:
            actor = actors[side]
            if not actor.alive:
                continue
            skill, wait_until = _choose_action(actor, policies[side], current_time)
            if skill is None:
                wake_at = max(current_time + 1e-6, wait_until or current_time + 0.05)
                heapq.heappush(event_heap, (wake_at, next(sequence), "ready", side, None))
                continue
            actor.resource = max(0.0, actor.resource - skill.resource_cost)
            actor.resource_spent += skill.resource_cost
            actor.cooldowns[skill.id] = current_time + skill.cooldown
            impact_time = current_time + skill.cast_time
            ready_time = impact_time + skill.recovery
            heapq.heappush(event_heap, (impact_time, next(sequence), "impact", side, skill.id))
            heapq.heappush(event_heap, (ready_time, next(sequence), "ready", side, None))

    winner = scenario.player.id if outcome == "victory" else scenario.enemy.id if outcome == "defeat" else None
    return BattleResult(
        outcome=outcome,
        winner=winner,
        duration=round(min(end_time, scenario.formula.max_time), 6),
        player_hp=round(player.hp, 6),
        enemy_hp=round(enemy.hp, 6),
        player_damage=round(player.damage_dealt, 6),
        enemy_damage=round(enemy.damage_dealt, 6),
        player_overkill=round(player.overkill, 6),
        enemy_overkill=round(enemy.overkill, 6),
        player_resource_end=round(player.resource, 6),
        enemy_resource_end=round(enemy.resource, 6),
        player_resource_spent=round(player.resource_spent, 6),
        enemy_resource_spent=round(enemy.resource_spent, 6),
        events=event_log,
        skill_metrics={"player": player.skill_metrics, "enemy": enemy.skill_metrics},
        seed=int(seed),
    )
