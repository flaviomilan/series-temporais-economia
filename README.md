# Séries Temporais Econômicas — Companion didático

Projeto-estudo que percorre, etapa a etapa, a metodologia de análise de séries
temporais aplicada a **dados econômicos brasileiros reais**, casando a teoria
do livro de **Aileen Nielsen — _Análise de Séries Temporais_ (Practical Time
Series Analysis, O'Reilly)** com a formação em Economia.

- **Série principal:** IPCA (inflação mensal) — rica em sazonalidade.
- **Séries exógenas:** câmbio USD/BRL e Selic — para os capítulos multivariados.
- **Fonte:** API pública do SGS/Banco Central do Brasil (sem chave).
- **Stack:** Python 3.12, `uv`, `pandas`, `statsmodels`.

## Os dois tipos de correlação (a ideia central)

1. **Autocorrelação** — a série com ela mesma em defasagens passadas (ACF/PACF).
   É a base dos modelos AR/MA/ARIMA. Não exige outra variável.
2. **Correlação cruzada / multivariada** — o IPCA explicado por câmbio e Selic
   (VAR, causalidade de Granger, ARIMAX). É onde a teoria econômica entra.

O projeto cobre os dois: primeiro a estrutura interna da série, depois o ganho
preditivo de variáveis externas.

## Mapa capítulo (Nielsen) → módulo do projeto

| Fase | Tema | Capítulos (Nielsen) | Onde | Status |
|---|---|---|---|---|
| 0 | Fundação (ambiente, estrutura) | — | `pyproject.toml`, `src/` | ✅ |
| 1 | Carga + conhecendo os dados | 2, 5 | `src/series_eco/data/loaders.py`, `notebooks/01_dados.ipynb` | ✅ |
| 2 | EDA (decomposição, ACF/PACF) | 3 | `src/series_eco/eda/descriptive.py`, `eda/plots.py`, `notebooks/02_eda.ipynb` | ✅ |
| 3 | Estacionariedade (ADF/KPSS, diff) | 3, 6 | `src/series_eco/eda/stationarity.py`, `notebooks/03_stationarity.ipynb` | ✅ |
| 4 | ARIMA / SARIMA | 6 | `src/series_eco/models/arima.py`, `notebooks/04_arima.ipynb` | ✅ |
| 5 | Multivariado (VAR, Granger, ARIMAX) | 6 | `src/series_eco/models/multivariate.py`, `notebooks/05_multivariate.ipynb` | ✅ |
| 6 | Medição de erro e backtesting | 11 | `src/series_eco/eval/backtest.py`, `notebooks/06_backtesting.ipynb` | ✅ |
| 7 | ML (gradient boosting) | 8–9 | `src/series_eco/models/features.py`, `models/ml.py`, `notebooks/07_ml.ipynb` | ✅ |

## Estrutura

```
src/series_eco/
├── config.py        # códigos de séries (SGS) e caminhos
├── data/loaders.py  # download BCB + cache Parquet + painel mensal
├── eda/             # decomposição, ACF/PACF, testes de estacionariedade
├── models/          # ARIMA/SARIMA, VAR, baselines
└── eval/            # backtesting walk-forward e métricas
data/raw/            # snapshots Parquet das séries (cache reprodutível)
notebooks/           # um notebook por fase/capítulo
tests/               # pytest
```

## Como rodar

```bash
uv sync                 # instala dependências (Python 3.12 via uv)
uv run pytest -q        # roda os testes
uv run jupyter lab      # abre os notebooks
```

Exemplo de uso da carga de dados:

```python
from series_eco.data import loaders

ipca = loaders.fetch_series("ipca", start="2000-01-01")      # série única
painel = loaders.load_panel("2000-01-01")                    # IPCA + câmbio + Selic
```

## Dados

| Variável | Série SGS | Frequência original | Frequência no painel |
|---|---|---|---|
| IPCA (% a.m.) | 433 | mensal | mensal |
| Câmbio USD/BRL (venda) | 1 | diária | mensal (último do mês) |
| Selic (meta, % a.a.) | 4189 | diária | mensal (último do mês) |

Os snapshots em `data/raw/` são versionados para reprodutibilidade: as séries
do BCB podem sofrer revisão, então o cache fixa o estado usado na análise.
