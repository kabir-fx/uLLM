"""B5: the run conditions a session refuses to start without (decisions 0013 and 0016)."""

from pathlib import Path

from bench import conditions
from bench.conditions import parse_footprint, parse_power, parse_swap, violations
from bench.schema import Power, RunEnv

ON_BATTERY = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=22675555)\t39%; discharging; 2:53 remaining present: true
"""
ON_AC = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=22675555)\t100%; charged; 0:00 remaining present: true
"""
CHARGING = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=23199843)\t62%; charging; 1:10 remaining present: true
"""
NOT_CHARGING = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=23199843)\t80%; AC attached; not charging present: true
"""
DESKTOP = "Now drawing from 'AC Power'\n"
LID_OPEN = """      "AppleClamshellCausesSleep" = Yes
      "AppleClamshellState" = No
"""
SETTINGS = """System-wide power settings:
Currently in use:
 standby              1
 lowpowermode         0
 sleep                1
"""

GOOD = Power(on_ac=True, battery_percent=100, low_power_mode=False, charging=False, lid_open=True)


def test_pmset_and_ioreg_output_is_read_into_power() -> None:
    assert parse_power(ON_BATTERY, SETTINGS, LID_OPEN) == Power(
        on_ac=False, battery_percent=39, low_power_mode=False, charging=False, lid_open=True
    )
    assert parse_power(
        ON_AC, SETTINGS.replace("lowpowermode         0", "lowpowermode 1"), ""
    ) == Power(on_ac=True, battery_percent=100, low_power_mode=True, charging=False, lid_open=None)
    assert parse_power(CHARGING, SETTINGS, LID_OPEN.replace("= No", "= Yes")).charging is True
    assert parse_power(NOT_CHARGING, SETTINGS, "").charging is False
    assert parse_power(DESKTOP, "", "") == Power(
        on_ac=True, battery_percent=None, low_power_mode=None, charging=None, lid_open=None
    )
    assert parse_power(LID_OPEN.replace("= No", "= Yes"), "", "x").lid_open is None
    assert parse_power("", "", LID_OPEN.replace("= No", "= Yes")).lid_open is False


def test_a_session_under_b5_conditions_may_start() -> None:
    assert violations(GOOD, sudo=True, containers=0, powermetrics_running=False) == []
    no_daemon = violations(GOOD, sudo=True, containers=None, powermetrics_running=False)
    assert no_daemon == []


def test_each_broken_condition_is_named() -> None:
    bad = Power(
        on_ac=False, battery_percent=39, low_power_mode=True, charging=False, lid_open=False
    )
    found = violations(bad, sudo=False, containers=2, powermetrics_running=True)
    assert len(found) == 5
    for word in ("AC power", "Low Power Mode", "sudo -v", "containers", "powermetrics"):
        assert any(word in v for v in found), word


def test_an_unknown_power_state_is_not_taken_as_good() -> None:
    unknown = Power(
        on_ac=None, battery_percent=None, low_power_mode=None, charging=None, lid_open=None
    )
    assert len(violations(unknown, sudo=True, containers=0, powermetrics_running=False)) == 2


def test_swap_in_use_is_read_from_sysctl() -> None:
    assert parse_swap("total = 2048.00M  used = 740.75M  free = 1307.25M  (encrypted)") == int(
        740.75 * 2**20
    )
    assert parse_swap("total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)") == 0
    assert parse_swap("") is None


def test_footprint_and_its_peak_are_read_in_bytes() -> None:
    text = "    phys_footprint: 2474416 B\n    phys_footprint_peak: 3000000 B\n"
    assert parse_footprint(text) == (2474416, 3000000)
    assert parse_footprint("footprint: no process") is None


def test_main_records_the_machine_in_env_json(tmp_path: Path) -> None:
    out = tmp_path / "env.json"
    conditions.main([str(out)])
    env = RunEnv.model_validate_json(out.read_text())
    assert env.powermetrics is False
    assert env.started <= env.ended
