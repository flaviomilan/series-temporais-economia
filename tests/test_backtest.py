"""Testes de backtesting, métricas e Diebold-Mariano."""

from __future__ import annotations

import numpy as np
import pandas as pd

from series_eco.eval import backtest


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-01", periods=n, freq="MS")


def test_compute_metrics_zero_error_for_perfect_forecast():
    # Arrange
    actual = pd.Series([1.0, 2.0, 3.0, 4.0], index=_index(4))
    predicted = actual.copy()

    # Act
    m = backtest.compute_metrics(actual, predicted)

    # Assert
    assert m.mae == 0.0
    assert m.rmse == 0.0
    assert m.mape == 0.0


def test_walk_forward_naive_predicts_previous_value():
    # Arrange
    series = pd.Series(np.arange(1.0, 11.0), index=_index(10))

    # Act
    actual, pred = backtest.walk_forward(series, backtest.naive, n_test=3)

    # Assert: o naive prevê sempre o valor anterior ao alvo.
    assert list(actual) == [8.0, 9.0, 10.0]
    assert list(pred) == [7.0, 8.0, 9.0]


def test_diebold_mariano_flags_clearly_better_model():
    # Arrange: model1 erra pouco; model2 erra muito.
    n = 120
    rng = np.random.default_rng(11)
    actual = pd.Series(rng.normal(0, 1, n), index=_index(n))
    pred1 = actual + rng.normal(0, 0.1, n)
    pred2 = actual + rng.normal(0, 1.0, n)

    # Act
    dm = backtest.diebold_mariano(actual, pred1, pred2)

    # Assert
    assert dm.better == "model1"
    assert dm.pvalue < 0.05
