"""Modelo de ML para séries temporais (Nielsen, cap. 9).

Usamos gradient boosting (``HistGradientBoostingRegressor``) sobre as features
de :mod:`series_eco.models.features`. É a escolha adequada para o tamanho desta
amostra (~300 pontos mensais); redes neurais profundas (cap. 10) exigiriam
muito mais dados para não sobreajustar.

:func:`make_ml_forecaster` devolve um *forecaster* compatível com
:func:`series_eco.eval.backtest.walk_forward`, permitindo comparar ML e modelos
clássicos exatamente no mesmo backtest.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from series_eco.models import features

DEFAULT_RANDOM_STATE: int = 0


def make_ml_forecaster(
    n_lags: int = features.DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = features.DEFAULT_ROLL_WINDOWS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Callable[[pd.Series], float]:
    """Cria um forecaster de 1 passo baseado em gradient boosting.

    O modelo é re-treinado em cada chamada (a cada origem do walk-forward), o
    que mantém a avaliação honesta: nenhum dado futuro entra no treino.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    def forecaster(train: pd.Series) -> float:
        X, y = features.make_supervised(train, n_lags, roll_windows)
        model = HistGradientBoostingRegressor(random_state=random_state)
        model.fit(X, y)
        row = features.make_prediction_row(train, n_lags, roll_windows)
        return float(model.predict(row)[0])

    return forecaster
