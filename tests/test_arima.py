"""Testes dos modelos ARIMA/SARIMA."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from series_eco.models import arima


def _ar1(n: int = 300, phi: float = 0.6, seed: int = 7) -> pd.Series:
    """Processo AR(1) sintético: y_t = phi * y_{t-1} + ruído."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.Series(y, index=idx, name="ar1")


def test_fit_sarima_recovers_ar_coefficient():
    # Arrange
    series = _ar1(phi=0.6)

    # Act
    result = arima.fit_sarima(series, order=(1, 0, 0))

    # Assert: o coeficiente AR estimado fica próximo do verdadeiro (0.6).
    assert result.params["ar.L1"] == pytest.approx(0.6, abs=0.15)


def test_forecast_returns_requested_horizon():
    # Arrange
    result = arima.fit_sarima(_ar1(), order=(1, 0, 0))

    # Act
    mean, conf_int = arima.forecast(result, steps=6)

    # Assert
    assert len(mean) == 6
    assert conf_int.shape == (6, 2)


def test_residuals_of_correct_model_are_white_noise():
    # Arrange: modelo bem especificado para um AR(1).
    result = arima.fit_sarima(_ar1(), order=(1, 0, 0))

    # Act
    lb = arima.residual_ljung_box(result, lags=12)

    # Assert: não se rejeita independência ⇒ resíduos são ruído branco.
    assert lb.is_white_noise


def test_auto_select_returns_finite_aic_and_usable_model():
    # Arrange
    series = _ar1()

    # Act
    selected = arima.auto_select(series, d=0, max_p=2, max_q=2)

    # Assert
    assert math.isfinite(selected.aic)
    assert selected.order[1] == 0  # d não é varrido
    mean, _ = arima.forecast(selected.result, steps=3)
    assert len(mean) == 3
