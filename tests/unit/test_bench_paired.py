"""Paired ratios, their bootstrap interval over whole blocks, and the twin's noise floor
(decision 0016)."""

import pytest
from bench.paired import Estimate, bootstrap, claimed, floor, ratio


def test_the_median_pools_every_ratio_of_every_block() -> None:
    e = ratio({0: [1.0, 1.2], 1: [1.1], 2: [0.9, 1.3]}, seed=0)
    assert e.point == pytest.approx(1.1)
    assert e.prompts == 5
    assert e.blocks == 3


def test_the_interval_resamples_whole_blocks() -> None:
    # Three blocks, one ratio each: a resample's median is its middle block, and each of 1.0
    # and 1.2 is the middle one in 7 of 27 resamples, far more than 2.5%.
    e = ratio({0: [1.0], 1: [1.1], 2: [1.2]}, seed=0)
    assert (e.low, e.high) == (1.0, 1.2)


def test_any_statistic_gets_a_block_interval() -> None:
    e = bootstrap({b: [float(b)] for b in range(5)}, seed=0, stat=max)
    assert e.point == 4.0
    assert e.high == 4.0
    assert e.low is not None
    assert e.low < 4.0


def test_identical_blocks_give_an_interval_of_one_point() -> None:
    e = ratio({b: [0.98, 1.0, 1.02] for b in range(5)}, seed=0)
    assert (e.low, e.point, e.high) == (1.0, 1.0, 1.0)


def test_fewer_than_three_blocks_give_no_interval() -> None:
    e = ratio({0: [1.0, 1.1], 1: [1.2]}, seed=0)
    assert e.low is None
    assert e.high is None


def test_the_floor_is_the_twins_largest_distance_from_one() -> None:
    assert floor(Estimate(1.0, 0.99, 1.004, 36, 3)) == pytest.approx(0.01)
    assert floor(Estimate(1.0, None, None, 12, 1)) is None


def test_a_gap_inside_the_floor_is_not_claimed() -> None:
    assert claimed(Estimate(1.16, 1.15, 1.17, 84, 7), 0.03)
    assert not claimed(Estimate(1.02, 1.01, 1.03, 84, 7), 0.03)
    assert not claimed(Estimate(1.16, None, None, 24, 2), None)
