"""Gráficos didáticos reutilizáveis para análise de séries econômicas.

Cada função de plotagem retorna uma ``matplotlib.figure.Figure`` para poder ser
exibida em notebook ou salva em arquivo. As funções de *cálculo*
(:func:`accumulated_12m`, :func:`cross_correlation`) são separadas das de
plotagem, para serem testáveis sem backend gráfico.

A ideia é casar leitura econômica e estatística: anotar eventos macro relevantes
e, ao mesmo tempo, expor estrutura (sazonalidade, autocorrelação, dispersão).
"""

from __future__ import annotations

import pandas as pd

# Eventos macroeconômicos brasileiros para anotar nos gráficos de série longa.
ECONOMIC_EVENTS: dict[str, str] = {
    "1994-07-01": "Plano Real",
    "2008-09-01": "Crise global",
    "2015-01-01": "Recessão 2015–16",
    "2020-03-01": "COVID-19",
    "2021-03-01": "Aperto monetário",
}


def accumulated_12m(series: pd.Series) -> pd.Series:
    """Soma móvel de 12 meses.

    Para o IPCA (variação mensal), isso reconstrói a **inflação acumulada em 12
    meses** — o número de manchete que o noticiário reporta. Útil para conectar
    a série técnica ao indicador que o público conhece.
    """
    return series.rolling(12).sum()


def cross_correlation(
    target: pd.Series, other: pd.Series, maxlag: int = 12
) -> pd.Series:
    """Correlação cruzada: corr(target_t, other_{t-k}) para k em 0..maxlag.

    ``k > 0`` mede o quanto ``other`` *antecede* ``target``. Um pico em k indica
    que a variável explicativa lidera o alvo em k períodos.
    """
    joined = pd.concat([target, other], axis=1).dropna()
    t = joined.iloc[:, 0]
    o = joined.iloc[:, 1]
    values = {k: t.corr(o.shift(k)) for k in range(maxlag + 1)}
    return pd.Series(values, name="ccf")


def plot_series(
    series: pd.Series,
    title: str = "",
    ylabel: str = "",
    *,
    events: dict[str, str] | None = None,
    figsize: tuple[int, int] = (11, 4),
):
    """Plota a série completa, opcionalmente anotando eventos econômicos."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Data")

    if events:
        ymin, ymax = ax.get_ylim()
        for date_str, label in events.items():
            date = pd.Timestamp(date_str)
            if series.index.min() <= date <= series.index.max():
                ax.axvline(date, color="0.6", linestyle="--", linewidth=0.8)
                ax.text(date, ymax, f" {label}", rotation=90, va="top",
                        ha="left", fontsize=8, color="0.4")
    fig.tight_layout()
    return fig


def plot_histogram(series: pd.Series, title: str = "", bins: int = 30,
                   figsize: tuple[int, int] = (7, 4)):
    """Histograma da distribuição dos valores da série."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(series.dropna(), bins=bins, color="C0", alpha=0.8)
    ax.axvline(series.mean(), color="C1", linestyle="--", label=f"média={series.mean():.2f}")
    ax.set_title(title)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frequência")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_seasonal_subseries(series: pd.Series, figsize: tuple[int, int] = (10, 4)):
    """Boxplot da série por mês do ano — revela o padrão sazonal e sua dispersão."""
    import matplotlib.pyplot as plt

    by_month = [series[series.index.month == m].dropna() for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(by_month, tick_labels=[str(m) for m in range(1, 13)])
    ax.set_title("Distribuição por mês (sazonalidade)")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Valor")
    fig.tight_layout()
    return fig


def plot_rolling_stats(series: pd.Series, window: int = 12,
                      figsize: tuple[int, int] = (11, 4)):
    """Série com média e desvio-padrão móveis — diagnóstico visual de estacionariedade."""
    import matplotlib.pyplot as plt

    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series, color="0.7", linewidth=0.8, label="série")
    ax.plot(roll_mean.index, roll_mean, color="C0", label=f"média móvel ({window}m)")
    ax.fill_between(
        roll_std.index,
        roll_mean - roll_std,
        roll_mean + roll_std,
        color="C0",
        alpha=0.2,
        label="± 1 desvio móvel",
    )
    ax.set_title("Média e desvio móveis")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_lag(series: pd.Series, lag: int = 1, figsize: tuple[int, int] = (5, 5)):
    """Dispersão de y_t contra y_{t-lag} — a autocorrelação vista como nuvem de pontos."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(series.shift(lag), series, s=10, alpha=0.5)
    ax.set_title(f"Lag plot (defasagem {lag})")
    ax.set_xlabel(f"y(t-{lag})")
    ax.set_ylabel("y(t)")
    fig.tight_layout()
    return fig


def plot_cross_correlation(
    target: pd.Series, other: pd.Series, maxlag: int = 12,
    figsize: tuple[int, int] = (8, 4),
):
    """Gráfico de barras da correlação cruzada (ver :func:`cross_correlation`)."""
    import matplotlib.pyplot as plt

    ccf = cross_correlation(target, other, maxlag)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(ccf.index, ccf.to_numpy())
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_title(f"Correlação cruzada: {other.name} antecede {target.name}?")
    ax.set_xlabel("Defasagem k (meses)")
    ax.set_ylabel("corr(target_t, other_{t-k})")
    fig.tight_layout()
    return fig
