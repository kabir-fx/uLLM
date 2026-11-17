"""B5: the run conditions a full benchmark run refuses to start without (decision 0013)."""

from bench.conditions import Plan, parse_power, violations
from bench.schema import Power

ON_BATTERY = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=22675555)\t39%; discharging; 2:53 remaining present: true
"""
ON_AC = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=22675555)\t100%; charged; 0:00 remaining present: true
"""
DESKTOP = "Now drawing from 'AC Power'\n"
SETTINGS = """System-wide power settings:
Currently in use:
 standby              1
 lowpowermode         0
 sleep                1
"""

GOOD = Power(on_ac=True, battery_percent=100, low_power_mode=False)
B5 = Plan(repeats=3, warmup=4, cooldown_s=120)


def test_pmset_output_is_read_into_power() -> None:
    assert parse_power(ON_BATTERY, SETTINGS) == Power(
        on_ac=False, battery_percent=39, low_power_mode=False
    )
    assert parse_power(
        ON_AC, SETTINGS.replace("lowpowermode         0", "lowpowermode 1")
    ) == Power(on_ac=True, battery_percent=100, low_power_mode=True)
    assert parse_power(DESKTOP, "") == Power(on_ac=True, battery_percent=None, low_power_mode=None)


def test_a_run_under_b5_conditions_may_start() -> None:
    assert violations(GOOD, sudo=True, containers=0, plan=B5) == []
    assert violations(GOOD, sudo=True, containers=None, plan=B5) == []  # no Docker daemon


def test_each_broken_condition_is_named() -> None:
    bad = Power(on_ac=False, battery_percent=39, low_power_mode=True)
    found = violations(bad, sudo=False, containers=2, plan=Plan(repeats=1, warmup=0, cooldown_s=0))
    assert len(found) == 7
    for word in (
        "AC power",
        "Low Power Mode",
        "sudo -v",
        "containers",
        "repeats",
        "warm-up",
        "cooldown",
    ):
        assert any(word in v for v in found), word


def test_an_unknown_power_state_is_not_taken_as_good() -> None:
    unknown = Power(on_ac=None, battery_percent=None, low_power_mode=None)
    assert len(violations(unknown, sudo=True, containers=0, plan=B5)) == 2
