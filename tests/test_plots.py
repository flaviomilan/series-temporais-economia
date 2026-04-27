"""Testes da biblioteca de gráficos didáticos.

As funções de cálculo são testadas por valor; as de plotagem, por contrato
(retornam uma Figure), com backend não interativo.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from series_eco.eda import plots  # noqa: E402


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-01", periods=n, freq="MS")


def test_accumulated_12m_sums_trailing_year():
    # Arrange: variação constante de 1.0 ao mês.
    series = pd.Series(np.ones(24), index=_index(24), name="ipca")

    # Act
    acc = plots.accumulated_12m(series)

    # Assert: a partir do 12º mês, a soma móvel de 12 meses é 12.
    assert np.isnan(acc.iloc[:11]).all()
    assert acc.iloc[11] == 12.0
    assert acc.iloc[-1] == 12.0


def test_cross_correlation_peaks_at_constructed_lag():
    # Arrange: target_t = other_{t-3} ⇒ pico de correlação em k=3.
    n = 200
    rng = np.random.default_rng(4)
    other = pd.Series(rng.normal(0, 1, n), index=_index(n), name="other")
    target = other.shift(3).rename("target")

    # Act
    ccf = plots.cross_correlation(target, other, maxlag=6)

    # Assert
    assert ccf.idxmax() == 3
    assert ccf.loc[3] > 0.9


def test_plot_functions_return_figures():
    # Arrange
    idx = _index(60)
    series = pd.Series(np.sin(np.arange(60)) + 5, index=idx, name="s")
    other = pd.Series(np.cos(np.arange(60)), index=idx, name="o")

    # Act / Assert: cada função produz uma Figure.
    assert isinstance(plots.plot_series(series, "t", "y", events=plots.ECONOMIC_EVENTS), Figure)
    assert isinstance(plots.plot_histogram(series), Figure)
    assert isinstance(plots.plot_seasonal_subseries(series), Figure)
    assert isinstance(plots.plot_rolling_stats(series), Figure)
    assert isinstance(plots.plot_lag(series, lag=1), Figure)
    assert isinstance(plots.plot_cross_correlation(series, other), Figure)
