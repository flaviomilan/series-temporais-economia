"""Modelos ARIMA/SARIMA (Box-Jenkins) sobre statsmodels SARIMAX.

Fluxo Box-Jenkins (Nielsen, cap. 6):

1. **Identificação** — ordens (p, d, q) e sazonais (P, D, Q, s) a partir de
   ACF/PACF (Fase 2) e da ordem de integração (Fase 3).
2. **Estimação** — :func:`fit_sarima`.
3. **Diagnóstico** — :func:`residual_ljung_box`: resíduos devem ser ruído branco.
4. **Previsão** — :func:`forecast`.

:func:`auto_select` faz uma busca em grade por AIC quando não se quer fixar as
ordens manualmente.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

NO_SEASONAL: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True)
class LjungBoxResult:
    """Diagnóstico de autocorrelação dos resíduos.

    ``is_white_noise`` é ``True`` quando NÃO se rejeita a hipótese de
    independência (p-valor > alpha) — ou seja, o modelo capturou a estrutura.
    """

    lags: int
    min_pvalue: float
    is_white_noise: bool


@dataclass(frozen=True)
class SelectedModel:
    """Melhor modelo encontrado na busca em grade."""

    result: object
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float


def fit_sarima(
    series: pd.Series,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = NO_SEASONAL,
    trend: str | None = None,
):
    """Estima um SARIMAX. Avisos de convergência são silenciados."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def residual_ljung_box(
    result, lags: int = 12, alpha: float = 0.05
) -> LjungBoxResult:
    """Teste de Ljung-Box nos resíduos do modelo ajustado."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb = acorr_ljungbox(result.resid, lags=[lags], return_df=True)
    pvalue = float(lb["lb_pvalue"].iloc[-1])
    return LjungBoxResult(lags=lags, min_pvalue=pvalue, is_white_noise=pvalue > alpha)


def auto_select(
    series: pd.Series,
    *,
    d: int = 0,
    max_p: int = 2,
    max_q: int = 2,
    seasonal: bool = False,
    seasonal_period: int = 12,
    seasonal_diff: int = 0,
    max_seasonal_p: int = 1,
    max_seasonal_q: int = 1,
) -> SelectedModel:
    """Busca em grade pelo menor AIC.

    A ordem de integração ``d`` (e ``seasonal_diff``) vem da análise da Fase 3,
    não é varrida — diferenciar dentro da busca por AIC compararia modelos com
    alvos diferentes, o que invalida a comparação de AIC.
    """
    seasonal_p_range = range(max_seasonal_p + 1) if seasonal else [0]
    seasonal_q_range = range(max_seasonal_q + 1) if seasonal else [0]

    best: SelectedModel | None = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for sp in seasonal_p_range:
                for sq in seasonal_q_range:
                    order = (p, d, q)
                    sorder = (
                        (sp, seasonal_diff, sq, seasonal_period)
                        if seasonal
                        else NO_SEASONAL
                    )
                    try:
                        res = fit_sarima(series, order, sorder)
                    except Exception:  # combinações que não convergem
                        continue
                    if best is None or res.aic < best.aic:
                        best = SelectedModel(res, order, sorder, float(res.aic))

    if best is None:
        raise RuntimeError("Nenhum modelo convergiu na grade especificada.")
    return best


def forecast(result, steps: int = 12, alpha: float = 0.05):
    """Previsão fora da amostra. Retorna ``(média, intervalo_confiança)``."""
    fc = result.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int(alpha=alpha)
