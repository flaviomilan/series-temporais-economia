"""Backtesting de origem móvel, métricas e teste de Diebold-Mariano.

Avaliar previsão *fora da amostra* é o que separa um modelo que decorou o
passado de um que generaliza (Nielsen, cap. 11). Aqui:

- :func:`walk_forward` — re-treina e prevê 1 passo à frente em janela expansiva.
- baselines :func:`naive` e :func:`seasonal_naive` — o piso que um modelo
  precisa superar.
- :func:`compute_metrics` — MAE, RMSE, MAPE.
- :func:`diebold_mariano` — a diferença de acurácia entre dois modelos é
  estatisticamente significativa?
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Um forecaster recebe a série de treino e devolve a previsão de 1 passo.
Forecaster = Callable[[pd.Series], float]


@dataclass(frozen=True)
class Metrics:
    """Métricas de erro de previsão."""

    mae: float
    rmse: float
    mape: float


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Resultado do teste de Diebold-Mariano (H0: mesma acurácia)."""

    statistic: float
    pvalue: float
    better: str  # "model1", "model2" ou "empate"


def naive(train: pd.Series) -> float:
    """Baseline: repete o último valor observado."""
    return float(train.iloc[-1])


def seasonal_naive(train: pd.Series, period: int = 12) -> float:
    """Baseline sazonal: repete o valor de ``period`` passos atrás."""
    return float(train.iloc[-period])


def walk_forward(
    series: pd.Series,
    forecaster: Forecaster,
    n_test: int,
) -> tuple[pd.Series, pd.Series]:
    """Validação de origem móvel, 1 passo à frente, com janela expansiva.

    Para cada um dos últimos ``n_test`` pontos, treina em tudo que veio antes e
    prevê o ponto seguinte. Re-treinar a cada passo evita vazamento de dados do
    futuro para o passado.

    Returns:
        ``(reais, previstos)`` — duas séries alinhadas pelo índice de datas.
    """
    n = len(series)
    if not 0 < n_test < n:
        raise ValueError(f"n_test deve estar em (0, {n}); recebido {n_test}.")

    start = n - n_test
    actuals: list[float] = []
    preds: list[float] = []
    for i in range(start, n):
        train = series.iloc[:i]
        preds.append(float(forecaster(train)))
        actuals.append(float(series.iloc[i]))

    idx = series.index[start:n]
    return (
        pd.Series(actuals, index=idx, name="real"),
        pd.Series(preds, index=idx, name="previsto"),
    )


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> Metrics:
    """Calcula MAE, RMSE e MAPE.

    O MAPE ignora pontos com valor real igual a zero (indefinido nesse caso) —
    relevante para o IPCA, que pode zerar ou ficar negativo.
    """
    a = actual.to_numpy(dtype=float)
    p = predicted.to_numpy(dtype=float)
    err = a - p

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    nonzero = a != 0
    mape = (
        float(np.mean(np.abs(err[nonzero] / a[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return Metrics(mae=mae, rmse=rmse, mape=mape)


def diebold_mariano(
    actual: pd.Series,
    pred1: pd.Series,
    pred2: pd.Series,
    loss: str = "squared",
    horizon: int = 1,
    alpha: float = 0.05,
) -> DieboldMarianoResult:
    """Teste de Diebold-Mariano comparando a acurácia de dois modelos.

    H0: os dois têm a mesma acurácia preditiva. ``better`` indica qual modelo
    tem menor perda *somente* quando a diferença é significativa (p < alpha);
    caso contrário, ``"empate"``.
    """
    from scipy.stats import norm

    e1 = (actual - pred1).to_numpy(dtype=float)
    e2 = (actual - pred2).to_numpy(dtype=float)

    if loss == "squared":
        d = e1**2 - e2**2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss deve ser 'squared' ou 'absolute'.")

    n = len(d)
    d_bar = float(np.mean(d))

    # Variância de longo prazo (HAC) acumulando autocovariâncias até horizon-1.
    var = float(np.var(d, ddof=0))
    for k in range(1, horizon):
        gamma_k = float(np.cov(d[k:], d[:-k])[0, 1])
        var += 2 * gamma_k

    statistic = d_bar / np.sqrt(var / n)
    pvalue = float(2 * (1 - norm.cdf(abs(statistic))))

    if pvalue >= alpha:
        better = "empate"
    else:
        better = "model1" if d_bar < 0 else "model2"

    return DieboldMarianoResult(statistic=float(statistic), pvalue=pvalue, better=better)
