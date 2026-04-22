"""Análise exploratória descritiva de séries temporais.

Reúne as duas lentes da EDA clássica (Nielsen, cap. 3):

- **Decomposição** da série em tendência, sazonalidade e resíduo (STL).
- **Autocorrelação** (ACF/PACF) — a série correlacionada com suas próprias
  defasagens, que é o ponto de partida para identificar modelos AR/MA/ARIMA.

As funções de cálculo são separadas das de plotagem, para que a lógica seja
testável sem depender de backend gráfico.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

# Período sazonal padrão para séries mensais (12 meses no ano).
DEFAULT_SEASONAL_PERIOD: int = 12
DEFAULT_NLAGS: int = 24


@dataclass(frozen=True)
class Autocorrelation:
    """Valores de ACF e PACF de uma série, com a defasagem correspondente.

    ``acf[0]`` é sempre 1.0 (correlação da série consigo mesma na defasagem 0).
    """

    lags: np.ndarray
    acf: np.ndarray
    pacf: np.ndarray


def decompose_stl(
    series: pd.Series,
    period: int = DEFAULT_SEASONAL_PERIOD,
    *,
    robust: bool = True,
):
    """Decompõe a série em tendência, sazonalidade e resíduo via STL.

    STL (Seasonal-Trend decomposition using LOESS) é robusto a outliers e não
    assume sazonalidade rígida, sendo adequado a séries macroeconômicas.

    Args:
        series: série temporal com índice de datas.
        period: período sazonal (12 para dados mensais).
        robust: se ``True``, usa pesos robustos contra outliers.

    Returns:
        Resultado STL do ``statsmodels`` com atributos ``.trend``,
        ``.seasonal``, ``.resid`` e ``.observed``.

    Raises:
        ValueError: se a série tiver menos de dois períodos sazonais completos.
    """
    if len(series) < 2 * period:
        raise ValueError(
            f"STL exige ao menos {2 * period} observações para period={period}; "
            f"a série tem {len(series)}."
        )
    return STL(series, period=period, robust=robust).fit()


def autocorrelation(
    series: pd.Series,
    nlags: int = DEFAULT_NLAGS,
) -> Autocorrelation:
    """Calcula ACF e PACF da série até ``nlags`` defasagens.

    Args:
        series: série temporal (idealmente já estacionária).
        nlags: número de defasagens.

    Returns:
        :class:`Autocorrelation` com arrays de tamanho ``nlags + 1``.
    """
    values = series.to_numpy(dtype=float)
    acf_vals = acf(values, nlags=nlags, fft=True)
    pacf_vals = pacf(values, nlags=nlags)
    lags = np.arange(nlags + 1)
    return Autocorrelation(lags=lags, acf=acf_vals, pacf=pacf_vals)


def plot_decomposition(result, figsize: tuple[int, int] = (10, 8)):
    """Plota os quatro painéis da decomposição STL. Retorna a Figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    panels = [
        ("Observado", result.observed),
        ("Tendência", result.trend),
        ("Sazonalidade", result.seasonal),
        ("Resíduo", result.resid),
    ]
    for ax, (title, comp) in zip(axes, panels):
        ax.plot(comp)
        ax.set_ylabel(title)
    axes[-1].set_xlabel("Data")
    fig.tight_layout()
    return fig


def plot_acf_pacf(
    series: pd.Series,
    nlags: int = DEFAULT_NLAGS,
    figsize: tuple[int, int] = (10, 4),
):
    """Plota ACF e PACF lado a lado. Retorna a Figure."""
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=figsize)
    plot_acf(series, lags=nlags, ax=ax_acf)
    plot_pacf(series, lags=nlags, ax=ax_pacf)
    ax_acf.set_title("ACF")
    ax_pacf.set_title("PACF")
    fig.tight_layout()
    return fig
