"""Testes da engenharia de features e do forecaster de ML."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from series_eco.models import features, ml


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-01", periods=n, freq="MS")


def test_make_supervised_builds_correct_lags_and_columns():
    # Arrange
    series = pd.Series(np.arange(1.0, 31.0), index=_index(30), name="y")

    # Act
    X, y = features.make_supervised(series, n_lags=3, roll_windows=(3,))

    # Assert: colunas na ordem esperada e lag_1 = valor do mês anterior.
    assert list(X.columns) == features.feature_names(3, (3,))
    assert not X.isna().any().any()
    # Para o alvo y_t, lag_1 deve ser y_{t-1}.
    assert (X["lag_1"].to_numpy() == (y.to_numpy() - 1)).all()


def test_make_prediction_row_matches_training_columns():
    # Arrange
    series = pd.Series(np.arange(1.0, 31.0), index=_index(30), name="y")

    # Act
    row = features.make_prediction_row(series, n_lags=3, roll_windows=(3,))

    # Assert
    assert list(row.columns) == features.feature_names(3, (3,))
    # 3 lags + rollmean_3 + rollstd_3 + month = 6 colunas.
    assert row.shape == (1, 6)
    assert row["lag_1"].iloc[0] == 30.0  # último valor observado


def test_ml_forecaster_predicts_seasonal_pattern():
    # Arrange: série cujo valor é exatamente o número do mês (sazonal puro).
    idx = _index(120)
    series = pd.Series(idx.month.astype(float), index=idx, name="y")
    forecaster = ml.make_ml_forecaster(n_lags=12, roll_windows=(3,))

    # Act: prever o mês seguinte ao fim da série.
    pred = forecaster(series)
    true_next = (idx[-1] + pd.offsets.MonthBegin(1)).month

    # Assert: o modelo recupera o padrão sazonal de forma aproximada.
    assert math.isfinite(pred)
    assert abs(pred - true_next) < 1.5
