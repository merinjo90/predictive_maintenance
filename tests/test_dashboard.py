"""Tests for the pure helper functions in app/dashboard.py."""

import numpy as np
import pandas as pd
import pytest

from app.dashboard import (
    SENSOR_COLUMNS,
    TYPE_COLUMNS,
    metrics_at_threshold,
    prepare_input,
    risk_level,
    threshold_sweep,
)


def _raw_rows(types=("L", "M")):
    return pd.DataFrame({
        "UDI": range(len(types)),
        "Type": list(types),
        "Air temperature [K]": [298.1] * len(types),
        "Process temperature [K]": [308.6] * len(types),
        "Rotational speed [rpm]": [1551] * len(types),
        "Torque [Nm]": [42.8] * len(types),
        "Tool wear [min]": [0] * len(types),
    })


def test_prepare_input_creates_all_type_columns():
    out = prepare_input(_raw_rows(types=("L", "L")))
    assert list(out.columns) == SENSOR_COLUMNS + TYPE_COLUMNS
    assert out["Type_L"].all()
    assert not out["Type_H"].any() and not out["Type_M"].any()


def test_prepare_input_accepts_encoded_columns_and_drops_extras():
    df = _raw_rows().drop(columns="Type")
    df["Type_H"], df["Type_L"], df["Type_M"] = [0, 0], [1, 0], [0, 1]
    out = prepare_input(df)
    assert "UDI" not in out.columns
    assert out["Type_M"].tolist() == [False, True]


def test_prepare_input_rejects_missing_column():
    with pytest.raises(ValueError, match="Torque"):
        prepare_input(_raw_rows().drop(columns="Torque [Nm]"))


def test_prepare_input_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown Type"):
        prepare_input(_raw_rows(types=("L", "X")))


def test_risk_level_bands():
    assert risk_level(0.30, 0.30) == "Failure predicted"
    assert risk_level(0.20, 0.30) == "Watch"
    assert risk_level(0.10, 0.30) == "Normal"


def test_metrics_at_threshold_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = metrics_at_threshold(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5


def test_threshold_sweep_shape():
    sweep = threshold_sweep([0, 1], [0.2, 0.8], thresholds=[0.1, 0.5, 0.9])
    assert len(sweep) == 3
    assert sweep.loc[1, "f1"] == 1.0
