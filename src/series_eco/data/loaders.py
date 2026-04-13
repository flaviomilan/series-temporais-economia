"""Carga de séries econômicas do SGS/BCB com cache local em Parquet.

Estratégia de reprodutibilidade: cada série baixada da API é persistida como
snapshot Parquet em ``data/raw/``. Execuções seguintes leem o cache, de modo que
notebooks e testes não dependem da disponibilidade da API nem de revisões
posteriores das séries.

Uso típico::

    from series_eco.data import loaders

    ipca = loaders.fetch_series("ipca", start="2000-01-01")
    painel = loaders.load_panel(start="2000-01-01")
"""

from __future__ import annotations

import pandas as pd

from series_eco.config import RAW_DIR, SGS_SERIES, TARGET_FREQ

# Início fixo do histórico baixado (Plano Real). O snapshot guarda a série
# inteira; o recorte por data é aplicado na LEITURA, não no download. Assim o
# cache não depende da janela pedida na primeira chamada (evita a falha
# silenciosa de devolver a janela errada quando o cache já existe).
HISTORY_START: str = "1994-07-01"

# Como cada série é reamostrada para a frequência mensal-alvo.
# IPCA já é mensal; câmbio (diário) usa o último valor do mês; Selic usa a
# meta vigente no fim do mês.
_RESAMPLE_AGG: dict[str, str] = {
    "ipca": "last",
    "cambio": "last",
    "selic": "last",
}


def _cache_path(name: str) -> "object":
    """Caminho do snapshot Parquet para uma série nomeada."""
    return RAW_DIR / f"{name}.parquet"


def _download_series(name: str, start: str, end: str | None) -> pd.Series:
    """Baixa uma única série do SGS via ``python-bcb``.

    Isolado em função própria para permitir mock em testes sem rede.
    """
    from bcb import sgs  # import tardio: evita custo de import quando há cache

    code = SGS_SERIES[name]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    # O SGS limita séries diárias a janelas de 10 anos por consulta. Baixamos em
    # blocos de até 10 anos e concatenamos — funciona também para mensais.
    chunks: list[pd.Series] = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(
            cursor + pd.DateOffset(years=10) - pd.DateOffset(days=1), end_ts
        )
        frame = sgs.get(
            {name: code},
            start=cursor.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
        )
        if not frame.empty:
            chunks.append(frame[name])
        cursor = chunk_end + pd.DateOffset(days=1)

    series = pd.concat(chunks) if chunks else pd.Series(dtype=float, name=name)
    series = series[~series.index.duplicated(keep="first")]
    series.name = name
    return series


def fetch_series(
    name: str,
    start: str,
    end: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.Series:
    """Retorna uma série econômica, baixando do BCB ou lendo o cache.

    Args:
        name: chave da série em ``SGS_SERIES`` (ex.: ``"ipca"``).
        start: data inicial no formato ``"YYYY-MM-DD"``.
        end: data final opcional ``"YYYY-MM-DD"``; ``None`` busca até hoje.
        use_cache: se ``True``, lê/grava snapshot Parquet em ``data/raw/``.

    Returns:
        ``pd.Series`` indexada por data, nomeada como ``name``.

    Raises:
        KeyError: se ``name`` não estiver registrado em ``SGS_SERIES``.
    """
    if name not in SGS_SERIES:
        raise KeyError(
            f"Série desconhecida: {name!r}. Disponíveis: {sorted(SGS_SERIES)}"
        )

    path = _cache_path(name)
    if use_cache and path.exists():
        full = pd.read_parquet(path)[name]
    else:
        # Baixa o histórico completo (não a janela pedida) para que o snapshot
        # sirva a qualquer recorte posterior.
        full = _download_series(name, start=HISTORY_START, end=None)
        if use_cache:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            full.to_frame().to_parquet(path)

    full.name = name
    # O recorte por data é aplicado na leitura, sobre o histórico completo.
    return full.loc[start:end]


def load_panel(
    start: str,
    end: str | None = None,
    *,
    names: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Monta um painel mensal alinhado com IPCA e variáveis exógenas.

    Cada série é reamostrada para a frequência-alvo (``TARGET_FREQ``) e unida
    pelo índice de datas. Linhas com qualquer valor ausente após o alinhamento
    são descartadas, de modo que o painel resultante é pronto para os modelos
    multivariados (VAR/ARIMAX).

    Args:
        start: data inicial ``"YYYY-MM-DD"``.
        end: data final opcional ``"YYYY-MM-DD"``.
        names: subconjunto de séries; padrão são todas em ``SGS_SERIES``.
        use_cache: repassado a :func:`fetch_series`.

    Returns:
        ``pd.DataFrame`` mensal, uma coluna por série, sem linhas incompletas.
    """
    selected = names if names is not None else list(SGS_SERIES)

    columns: dict[str, pd.Series] = {}
    for name in selected:
        raw = fetch_series(name, start=start, end=end, use_cache=use_cache)
        agg = _RESAMPLE_AGG.get(name, "last")
        monthly = raw.resample(TARGET_FREQ).agg(agg)
        columns[name] = monthly

    panel = pd.DataFrame(columns)
    return panel.dropna(how="any")
