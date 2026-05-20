"""Análise multivariada: Granger, VAR e ARIMAX.

Onde a teoria econômica entra (Nielsen, cap. 6): em vez de explicar o IPCA só
pela sua própria história (autocorrelação), perguntamos se **outras** variáveis
— câmbio, Selic — ajudam a explicá-lo (correlação cruzada).

- :func:`granger_causality` — câmbio/Selic "Granger-causam" o IPCA?
- :func:`fit_var` — sistema de equações onde todas as séries se influenciam.
- :func:`fit_arimax` — SARIMA do IPCA com variáveis exógenas.

**Atenção:** Granger e VAR pressupõem séries estacionárias. Câmbio e Selic em
nível têm raiz unitária; use as séries diferenciadas (ver Fase 3).
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX

from series_eco.models.arima import NO_SEASONAL


@dataclass(frozen=True)
class GrangerResult:
    """Resultado do teste de causalidade de Granger.

    ``causes`` é ``True`` quando ``cause`` ajuda a prever ``target`` (p-valor
    mínimo entre as defasagens abaixo de ``alpha``).
    """

    best_lag: int
    min_pvalue: float
    causes: bool


def granger_causality(
    target: pd.Series,
    cause: pd.Series,
    maxlag: int = 12,
    alpha: float = 0.05,
) -> GrangerResult:
    """Testa se ``cause`` Granger-causa ``target`` em até ``maxlag`` defasagens."""
    from statsmodels.tsa.stattools import grangercausalitytests

    data = pd.concat([target, cause], axis=1).dropna()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # redirect_stdout: a função imprime tabelas por padrão em algumas versões.
        with contextlib.redirect_stdout(io.StringIO()):
            res = grangercausalitytests(data, maxlag=maxlag)

    pvalues = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
    best_lag = min(pvalues, key=pvalues.get)
    min_pvalue = float(pvalues[best_lag])
    return GrangerResult(best_lag=best_lag, min_pvalue=min_pvalue, causes=min_pvalue < alpha)


def fit_var(panel: pd.DataFrame, maxlags: int = 12, ic: str = "aic"):
    """Ajusta um VAR escolhendo a ordem pelo critério ``ic`` (padrão AIC).

    O ``panel`` deve conter apenas séries estacionárias.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VAR(panel.dropna()).fit(maxlags=maxlags, ic=ic)


def fit_arimax(
    endog: pd.Series,
    exog: pd.DataFrame,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = NO_SEASONAL,
):
    """Ajusta um SARIMAX com variáveis exógenas (ARIMAX/SARIMAX).

    ``endog`` e ``exog`` devem compartilhar o mesmo índice; o alinhamento é
    feito por interseção das datas.
    """
    aligned = pd.concat([endog, exog], axis=1).dropna()
    y = aligned[endog.name]
    x = aligned[list(exog.columns)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y,
            exog=x,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def impulse_response(var_result, periods: int = 12):
    """Funções de resposta a impulso (IRF) de um VAR ajustado.

    Mostram como um choque de 1 desvio-padrão numa variável se propaga pelas
    demais ao longo de ``periods`` meses — a visualização-assinatura do VAR.
    Retorna o objeto IRF do statsmodels (use ``.plot(...)`` ou ``.irfs``).
    """
    return var_result.irf(periods)
