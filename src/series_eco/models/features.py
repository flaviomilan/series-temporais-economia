"""Engenharia de features para séries temporais (Nielsen, cap. 8).

Modelos de ML não entendem "tempo" — precisam de uma tabela `X, y` onde cada
linha descreve o estado passado da série. Geramos:

- **defasagens** (`lag_1 ... lag_n`): os valores dos meses anteriores;
- **estatísticas móveis** (média e desvio) sobre janelas, usando só o passado;
- **mês** (1–12): deixa o modelo aprender a sazonalidade sem dummies explícitas.

As mesmas colunas, na mesma ordem, são produzidas para o treino
(:func:`make_supervised`) e para a previsão de 1 passo (:func:`make_prediction_row`).
"""

from __future__ import annotations

import pandas as pd

DEFAULT_LAGS: int = 12
DEFAULT_ROLL_WINDOWS: tuple[int, ...] = (3, 12)


def feature_names(n_lags: int, roll_windows: tuple[int, ...]) -> list[str]:
    """Nomes das colunas de feature, em ordem fixa e reproduzível."""
    names = [f"lag_{lag}" for lag in range(1, n_lags + 1)]
    for window in roll_windows:
        names += [f"rollmean_{window}", f"rollstd_{window}"]
    names.append("month")
    return names


def make_supervised(
    series: pd.Series,
    n_lags: int = DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Transforma a série num par ``(X, y)`` supervisionado.

    As estatísticas móveis usam ``shift(1)`` — só informação até o mês anterior,
    sem vazar o valor que se quer prever.
    """
    df = pd.DataFrame(index=series.index)
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)

    shifted = series.shift(1)
    for window in roll_windows:
        df[f"rollmean_{window}"] = shifted.rolling(window).mean()
        df[f"rollstd_{window}"] = shifted.rolling(window).std()

    df["month"] = series.index.month
    df["y"] = series
    df = df.dropna()

    cols = feature_names(n_lags, roll_windows)
    return df[cols], df["y"]


def make_prediction_row(
    series: pd.Series,
    n_lags: int = DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
    target_month: int | None = None,
) -> pd.DataFrame:
    """Monta a linha de features para prever o mês imediatamente após ``series``.

    Espelha exatamente as colunas de :func:`make_supervised`, calculadas sobre a
    cauda da série de treino.
    """
    if target_month is None:
        target_month = (series.index[-1] + pd.offsets.MonthBegin(1)).month

    row: dict[str, float] = {}
    for lag in range(1, n_lags + 1):
        row[f"lag_{lag}"] = float(series.iloc[-lag])
    for window in roll_windows:
        tail = series.iloc[-window:]
        row[f"rollmean_{window}"] = float(tail.mean())
        row[f"rollstd_{window}"] = float(tail.std())
    row["month"] = float(target_month)

    cols = feature_names(n_lags, roll_windows)
    return pd.DataFrame([row], columns=cols)
