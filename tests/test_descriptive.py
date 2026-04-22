"""Testes da EDA descritiva (decomposição e autocorrelação)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from series_eco.eda import descriptive


def _seasonal_series(n: int = 48) -> pd.Series:
    """Série mensal sintética: tendência linear + sazonalidade anual + ruído."""
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = np.sin(2 * np.pi * t / 12)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, n)
    return pd.Series(trend + seasonal + noise, index=idx, name="y")


def test_decompose_stl_reconstructs_series():
    # Arrange
    series = _seasonal_series()

    # Act
    result = descriptive.decompose_stl(series, period=12)
    reconstructed = result.trend + result.seasonal + result.resid

    # Assert: a soma das componentes recompõe a série observada.
    np.testing.assert_allclose(reconstructed.to_numpy(), series.to_numpy(), atol=1e-8)


def test_decompose_stl_requires_two_full_periods():
    # Arrange: série curta demais (menos de 2*period).
    short = _seasonal_series(n=10)

    # Act / Assert
    with pytest.raises(ValueError):
        descriptive.decompose_stl(short, period=12)


def test_autocorrelation_shapes_and_lag_zero():
    # Arrange
    series = _seasonal_series()
    nlags = 18

    # Act
    result = descriptive.autocorrelation(series, nlags=nlags)

    # Assert
    assert result.acf.shape == (nlags + 1,)
    assert result.pacf.shape == (nlags + 1,)
    assert result.lags[0] == 0
    # ACF na defasagem 0 é sempre 1 (série consigo mesma).
    assert result.acf[0] == pytest.approx(1.0)
