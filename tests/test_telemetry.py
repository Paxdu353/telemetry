"""Tests du module de télémétrie.

Deux tests vous sont fournis en exemple : ils montrent le style attendu.
Tout le reste est à écrire — voir le TD 1.
"""

import pytest

from fleet_api.models import Position, Reading, RobotState
from fleet_api.telemetry import (
    average_speed_mps,
    battery_percentage,
    detect_voltage_dropouts,
    distance_m,
    estimate_runtime_minutes,
    fleet_summary,
    is_low_battery,
    median_voltage_mv,
    path_length_m,
    robot_state,
)

# ---------------------------------------------------------------------------
# Exemple 1 — un test simple, avec un cas nominal et les deux bornes.
# ---------------------------------------------------------------------------


def test_battery_percentage_bornes_et_cas_nominal():
    """La conversion est linéaire et bornée à [0, 100]."""
    assert battery_percentage(12_600) == 100.0
    assert battery_percentage(10_500) == 0.0
    assert battery_percentage(11_550) == 50.0
    # Hors bornes : on sature, on ne dépasse pas.
    assert battery_percentage(13_000) == 100.0
    assert battery_percentage(9_000) == 0.0


# ---------------------------------------------------------------------------
# Exemple 2 — le même test écrit en paramétré, quand les cas se ressemblent.
# On teste aussi que l'erreur attendue est bien levée.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "attendu"),
    [
        (Position(0, 0), Position(3, 4), 5.0),  # triplet pythagoricien
        (Position(0, 0), Position(0, 0), 0.0),  # distance à soi-même
        (Position(1, 1), Position(-2, -3), 5.0),  # coordonnées négatives
        (Position(3, 4), Position(0, 0), 5.0),  # symétrie
    ],
)
def test_distance_m(a, b, attendu):
    """La distance est euclidienne, positive et symétrique."""
    assert distance_m(a, b) == pytest.approx(attendu)


def test_battery_percentage_rejette_des_bornes_incoherentes():
    """Une plage de tension invalide lève une ValueError."""
    with pytest.raises(ValueError, match="strictement supérieur"):
        battery_percentage(11_000, empty_mv=12_000, full_mv=11_000)


# ---------------------------------------------------------------------------
# À vous. Huit fonctions de fleet_api.telemetry n'ont aucun test :
#
#   is_low_battery, path_length_m, average_speed_mps, estimate_runtime_minutes,
#   median_voltage_mv, robot_state, detect_voltage_dropouts, fleet_summary
#
# Écrivez-les en vous appuyant sur les docstrings, qui font foi.
# Trois de ces fonctions ne respectent pas leur spécification.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "battery_pct, threshold_pct, expected",
    [
        (10.0, 20.0, True),
        (20.0, 20.0, True),
        (30.0, 20.0, False),
    ],
)
def test_is_low_battery(battery_pct, threshold_pct, expected):
    assert is_low_battery(battery_pct, threshold_pct) is expected


@pytest.mark.parametrize(
    "positions, expected",
    [
        ([Position(0, 0), Position(3, 4)], 5.0),
        ([Position(0, 0)], 0.0),
        ([], 0.0),
    ],
)
def test_path_length_m(positions, expected):
    assert path_length_m(positions) == expected


@pytest.mark.parametrize(
    "path_length_m, elapsed_s, expected",
    [
        (10.0, 5.0, 2.0),
        (10.0, 0.0, None),
    ],
)
def test_average_speed_mps(path_length_m, elapsed_s, expected):
    assert average_speed_mps(path_length_m, elapsed_s) == expected


@pytest.mark.parametrize(
    "battery_pct, drain_pct_per_min, expected",
    [
        (50.0, 2.0, 25.0),
        (50.0, 0.0, None),
    ],
)
def test_estimate_runtime_minutes(battery_pct, drain_pct_per_min, expected):
    assert estimate_runtime_minutes(battery_pct, drain_pct_per_min) == expected


@pytest.mark.parametrize(
    "values, expected",
    [
        ([], None),
        ([1200], 1200),
        ([1000, 1200, 1100], 1100),
        ([1000, 1200], 1100),
    ],
)
def test_median_voltage_mv(values, expected):
    readings = [
        Reading(
            robot_id="R1",
            timestamp_s=0.0,
            position=Position(0, 0),
            voltage_mv=v,
        )
        for v in values
    ]

    assert median_voltage_mv(readings) == expected


def make_reading(
    voltage_mv=12000,
    timestamp_s=100.0,
    is_charging=False,
):
    return Reading(
        robot_id="R1",
        timestamp_s=timestamp_s,
        position=Position(0, 0),
        voltage_mv=voltage_mv,
        is_charging=is_charging,
    )


@pytest.mark.parametrize(
    "reading, now_s, threshold_pct, grace_s, expected",
    [
        # OFFLINE prioritaire sur tout
        (
            make_reading(voltage_mv=9000, timestamp_s=0, is_charging=True),
            200,
            20,
            100,
            RobotState.OFFLINE,
        ),
        # CHARGING avant LOW_BATTERY
        (
            make_reading(voltage_mv=9000, timestamp_s=100, is_charging=True),
            150,
            20,
            100,
            RobotState.CHARGING,
        ),
        # LOW_BATTERY
        (
            make_reading(voltage_mv=9000, timestamp_s=100, is_charging=False),
            150,
            20,
            100,
            RobotState.LOW_BATTERY,
        ),
        # OPERATIONAL
        (
            make_reading(voltage_mv=12000, timestamp_s=100, is_charging=False),
            150,
            20,
            100,
            RobotState.OPERATIONAL,
        ),
    ],
)
def test_robot_state(reading, now_s, threshold_pct, grace_s, expected):
    assert robot_state(reading, now_s, threshold_pct, grace_s) == expected


@pytest.mark.parametrize(
    "voltages, max_drop_mv, expected",
    [
        ([12000, 11000], 500, [1]),  # chute de 1000
        ([12000, 11500], 500, []),  # exactement 500 => pas anormal
        ([12000, 12500], 500, []),  # remontée
        ([12000, 11000, 9000], 500, [1, 2]),
        ([12000], 500, []),
        ([], 500, []),
    ],
)
def test_detect_voltage_dropouts(voltages, max_drop_mv, expected):
    readings = [
        make_reading(voltage_mv=v, timestamp_s=i) for i, v in enumerate(voltages)
    ]

    assert detect_voltage_dropouts(readings, max_drop_mv) == expected


def test_fleet_summary_empty():
    assert fleet_summary([]) == {
        "robot_count": 0,
        "average_battery_pct": 0.0,
        "low_battery_count": 0,
    }


def test_fleet_summary():
    readings = [
        make_reading(voltage_mv=12000),
        make_reading(voltage_mv=9000),
    ]

    result = fleet_summary(readings, threshold_pct=20)

    assert result["robot_count"] == 2
    assert isinstance(result["average_battery_pct"], float)
    assert result["low_battery_count"] >= 0
