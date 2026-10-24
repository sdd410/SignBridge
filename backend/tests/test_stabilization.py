"""Tests for Temporal Prediction Stabilization."""

import time
import pytest
from backend.app.inference.stabilization import TemporalStabilizer


def test_confidence_threshold_filtering():
    stabilizer = TemporalStabilizer(window_size=5, confidence_threshold=0.7)

    # Low confidence predictions should not pass
    res = stabilizer.process("A", 0.5)
    assert not res["passes_confidence"]
    assert not res["is_accepted"]
    assert res["stability_progress"] == 0.0


def test_stability_duration_and_acceptance():
    stabilizer = TemporalStabilizer(
        window_size=5,
        confidence_threshold=0.6,
        stability_duration_ms=100,
        repeat_lockout_ms=500
    )

    # First frame
    res1 = stabilizer.process("B", 0.9)
    assert res1["passes_confidence"]
    assert not res1["is_accepted"]  # Not yet stable

    # Wait for stability duration to pass
    time.sleep(0.12)
    res2 = stabilizer.process("B", 0.9)
    assert res2["is_stable"]
    assert res2["is_accepted"]
    assert res2["accepted_char"] == "B"


def test_repeat_lockout():
    stabilizer = TemporalStabilizer(
        window_size=3,
        confidence_threshold=0.6,
        stability_duration_ms=50,
        repeat_lockout_ms=300
    )

    # Establish stability and accept
    time.sleep(0.06)
    stabilizer.process("C", 0.95)
    time.sleep(0.06)
    res1 = stabilizer.process("C", 0.95)
    assert res1["is_accepted"]

    # Immediate next frame of same character should be suppressed by lockout
    res2 = stabilizer.process("C", 0.95)
    assert not res2["is_accepted"]

    # Different character should be accepted once established and stable
    # Feed D across the window to establish majority
    for _ in range(3):
        stabilizer.process("D", 0.95)
    
    # Wait for required stability duration
    time.sleep(0.06)
    res3 = stabilizer.process("D", 0.95)
    assert res3["is_accepted"]
    assert res3["accepted_char"] == "D"
