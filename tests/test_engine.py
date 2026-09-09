from __future__ import annotations

import pytest
from pydantic import ValidationError

from combat_balance.engine import simulate_once
from combat_balance.models import (
    BuffConfig,
    CharacterTemplate,
    FormulaConfig,
    PolicyConfig,
    ScenarioConfig,
    SkillConfig,
    StatBlock,
)


def character(
    actor_id: str,
    hp: float,
    attack: float,
    multiplier: float = 1.0,
    skills: list[SkillConfig] | None = None,
) -> CharacterTemplate:
    return CharacterTemplate(
        id=actor_id,
        name=actor_id,
        archetype="test",
        stats=StatBlock(attack=attack, defense=0, hp=hp),
        basic_attack=SkillConfig(
            id=f"{actor_id}_basic",
            name="basic",
            multiplier=multiplier,
            cooldown=0,
            cast_time=1,
            recovery=0,
            hit_rate=1,
            priority=999,
        ),
        skills=skills or [],
    )


def scenario(player: CharacterTemplate, enemy: CharacterTemplate, priority: list[str] | None = None, max_time: float = 10):
    policy = PolicyConfig(id="priority", name="priority", kind="priority", priority=priority or [])
    return ScenarioConfig(
        player=player,
        enemy=enemy,
        level=1,
        player_policy=policy,
        enemy_policy=policy,
        formula=FormulaConfig(defense_k_base=100, defense_k_per_level=10, max_time=max_time),
    )


def test_exact_ttk_and_overkill_are_separated():
    player = character("player", 1000, 100)
    enemy = character("enemy", 150, 1, multiplier=0)
    result = simulate_once(scenario(player, enemy), seed=1)
    assert result.outcome == "victory"
    assert result.duration == 2
    assert result.player_damage == 150
    assert result.player_overkill == 50
    assert result.player_dps == 75


def test_simultaneous_lethal_impacts_are_draw():
    result = simulate_once(
        scenario(character("player", 100, 100), character("enemy", 100, 100)),
        seed=2,
    )
    assert result.outcome == "draw"
    assert result.duration == 1
    assert result.player_hp == 0
    assert result.enemy_hp == 0


def test_buff_has_left_closed_right_open_duration():
    buff_skill = SkillConfig(
        id="focus",
        name="focus",
        multiplier=0,
        cooldown=10,
        cast_time=0.1,
        recovery=0.1,
        apply_buff=BuffConfig(
            id="focus_buff",
            name="focus",
            duration=2,
            modifiers={"attack_pct": 1.0},
        ),
    )
    player = character("player", 1000, 100, skills=[buff_skill])
    enemy = character("enemy", 250, 1, multiplier=0)
    result = simulate_once(scenario(player, enemy, ["focus", "player_basic"]), seed=3)
    basic_hits = [event for event in result.events if event["skill_id"] == "player_basic"]
    assert basic_hits[0]["time"] == 1.2
    assert basic_hits[0]["raw_damage"] == 200
    assert basic_hits[1]["time"] == 2.2
    assert basic_hits[1]["raw_damage"] == 100


def test_zero_damage_battle_times_out_without_infinite_loop():
    result = simulate_once(
        scenario(character("player", 100, 1, 0), character("enemy", 100, 1, 0), max_time=2),
        seed=4,
    )
    assert result.outcome == "timeout"
    assert result.duration == 2


def test_invalid_probability_is_rejected():
    with pytest.raises(ValidationError):
        StatBlock(attack=10, defense=0, hp=100, dodge=1.1)


def test_zero_duration_action_is_rejected():
    with pytest.raises(ValidationError):
        SkillConfig(id="bad", name="bad", cast_time=0, recovery=0)

