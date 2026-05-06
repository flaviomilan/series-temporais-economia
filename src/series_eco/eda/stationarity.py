"""Testes de estacionariedade e ordem de diferenciação.

A estacionariedade é pré-requisito de Box-Jenkins (ARIMA). Usamos dois testes
complementares, com hipóteses nulas *opostas* — combiná-los é mais informativo
que olhar um só (Nielsen, cap. 3 e 6):

- **ADF** (Augmented Dickey-Fuller): H0 = a série tem raiz unitária (NÃO é
  estacionária). p-valor < alpha ⇒ rejeita H0 ⇒ estacionária.
- **KPSS**: H0 = a série É estacionária. p-valor < alpha ⇒ rejeita H0 ⇒ NÃO
  estacionária.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

DEFAULT_ALPHA: float = 0.05
MAX_DIFF: int = 2


@dataclass(frozen=True)
class TestResult:
    """Resultado de um teste de estacionariedade."""

    name: str
    statistic: float
    pvalue: float
    is_stationary: bool


@dataclass(frozen=True)
class StationarityReport:
    """Veredito combinado de ADF + KPSS."""

    adf: TestResult
    kpss: TestResult
    verdict: str


def adf_test(series: pd.Series, alpha: float = DEFAULT_ALPHA) -> TestResult:
    """Teste ADF. Estacionária quando p-valor < ``alpha``."""
    stat, pvalue, *_ = adfuller(series.dropna().to_numpy(dtype=float))
    return TestResult("ADF", float(stat), float(pvalue), pvalue < alpha)


def kpss_test(
    series: pd.Series,
    alpha: float = DEFAULT_ALPHA,
    regression: str = "c",
) -> TestResult:
    """Teste KPSS. Estacionária quando p-valor >= ``alpha`` (H0 = estacionária).

    O ``statsmodels`` emite ``InterpolationWarning`` quando o p-valor cai fora da
    tabela; é esperado e silenciado aqui.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, *_ = kpss(
            series.dropna().to_numpy(dtype=float), regression=regression
        )
    return TestResult("KPSS", float(stat), float(pvalue), pvalue >= alpha)


def stationarity_report(
    series: pd.Series, alpha: float = DEFAULT_ALPHA
) -> StationarityReport:
    """Combina ADF e KPSS num veredito interpretável.

    Quatro casos clássicos:

    - ambos estacionários ⇒ ``"estacionária"``
    - ambos não estacionários ⇒ ``"não estacionária"``
    - ADF estacionário, KPSS não ⇒ ``"tendência-estacionária"`` (remover tendência)
    - ADF não, KPSS estacionário ⇒ ``"diferença-estacionária"`` (diferenciar)
    """
    adf = adf_test(series, alpha)
    kpss_res = kpss_test(series, alpha)

    if adf.is_stationary and kpss_res.is_stationary:
        verdict = "estacionária"
    elif not adf.is_stationary and not kpss_res.is_stationary:
        verdict = "não estacionária"
    elif adf.is_stationary and not kpss_res.is_stationary:
        verdict = "tendência-estacionária"
    else:
        verdict = "diferença-estacionária"

    return StationarityReport(adf=adf, kpss=kpss_res, verdict=verdict)


def ndiffs(
    series: pd.Series, alpha: float = DEFAULT_ALPHA, max_diff: int = MAX_DIFF
) -> int:
    """Número de diferenças necessárias para o ADF acusar estacionariedade.

    Retorna o menor ``d`` em ``0..max_diff`` cuja série diferenciada passa no
    ADF; se nenhum passar, retorna ``max_diff``. Equivale à ordem ``d`` do ARIMA.
    """
    current = series.dropna()
    for d in range(max_diff + 1):
        if adf_test(current, alpha).is_stationary:
            return d
        current = current.diff().dropna()
    return max_diff
