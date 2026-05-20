"""Testes da análise multivariada (Granger, VAR, ARIMAX)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from series_eco.models import multivariate


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-01", periods=n, freq="MS")


def test_granger_detects_constructed_cause():
    # Arrange: target depende de cause defasada ⇒ deve haver causalidade.
    n = 200
    rng = np.random.default_rng(3)
    cause = rng.normal(0, 1, n)
    target = np.zeros(n)
    for t in range(1, n):
        target[t] = 0.6 * cause[t - 1] + rng.normal(0, 0.3)
    idx = _index(n)

    # Act
    res = multivariate.granger_causality(
        pd.Series(target, index=idx, name="target"),
        pd.Series(cause, index=idx, name="cause"),
        maxlag=4,
    )

    # Assert
    assert res.causes
    assert 0.0 <= res.min_pvalue <= 1.0
    assert res.best_lag >= 1


def test_fit_var_selects_finite_order():
    # Arrange: VAR(1) bivariado estacionário.
    n = 300
    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, (n, 2))
    y = np.zeros((n, 2))
    a = np.array([[0.5, 0.1], [0.0, 0.4]])
    for t in range(1, n):
        y[t] = a @ y[t - 1] + e[t]
    panel = pd.DataFrame(y, index=_index(n), columns=["a", "b"])

    # Act
    result = multivariate.fit_var(panel, maxlags=4)

    # Assert
    assert result.k_ar >= 1
    assert math.isfinite(result.aic)


def test_fit_arimax_recovers_exogenous_coefficient():
    # Arrange: endog = 2 * exog + ruído.
    n = 200
    rng = np.random.default_rng(9)
    idx = _index(n)
    exog = pd.DataFrame({"x": rng.normal(0, 1, n)}, index=idx)
    endog = pd.Series(
        2.0 * exog["x"].to_numpy() + rng.normal(0, 0.3, n), index=idx, name="y"
    )

    # Act
    result = multivariate.fit_arimax(endog, exog, order=(1, 0, 0))

    # Assert: o coeficiente da exógena fica próximo de 2.0.
    assert result.params["x"] == pytest.approx(2.0, abs=0.3)


def test_impulse_response_has_expected_shape():
    # Arrange: VAR(1) bivariado.
    n = 300
    rng = np.random.default_rng(6)
    e = rng.normal(0, 1, (n, 2))
    y = np.zeros((n, 2))
    a = np.array([[0.5, 0.2], [0.0, 0.4]])
    for t in range(1, n):
        y[t] = a @ y[t - 1] + e[t]
    panel = pd.DataFrame(y, index=_index(n), columns=["a", "b"])
    var_result = multivariate.fit_var(panel, maxlags=2)

    # Act
    periods = 10
    irf = multivariate.impulse_response(var_result, periods=periods)

    # Assert: irfs tem shape (periods + 1, k, k) para k variáveis.
    assert irf.irfs.shape == (periods + 1, 2, 2)
