"""Testes dos diagnósticos de estacionariedade."""

from __future__ import annotations

import numpy as np
import pandas as pd

from series_eco.eda import stationarity


def _white_noise(n: int = 300) -> pd.Series:
    """Ruído branco: estacionário por construção."""
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0, 1, n), index=idx, name="wn")


def _random_walk(n: int = 300) -> pd.Series:
    """Passeio aleatório: não estacionário (raiz unitária), I(1)."""
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(1)
    return pd.Series(np.cumsum(rng.normal(0, 1, n)), index=idx, name="rw")


def test_white_noise_is_stationary():
    # Arrange
    series = _white_noise()

    # Act
    report = stationarity.stationarity_report(series)

    # Assert
    assert report.adf.is_stationary
    assert report.kpss.is_stationary
    assert report.verdict == "estacionária"
    assert stationarity.ndiffs(series) == 0


def test_random_walk_needs_one_difference():
    # Arrange
    series = _random_walk()

    # Act
    report = stationarity.stationarity_report(series)

    # Assert: ADF não rejeita a raiz unitária; diferenciar uma vez resolve.
    assert not report.adf.is_stationary
    assert stationarity.ndiffs(series) == 1
