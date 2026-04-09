"""Configuração central do projeto: códigos de séries e caminhos.

Centraliza os identificadores das séries do Sistema Gerenciador de Séries
Temporais (SGS) do Banco Central do Brasil e os diretórios de dados, para que
loaders, notebooks e testes referenciem uma única fonte da verdade.

Referência das séries SGS: https://www3.bcb.gov.br/sgspub/
"""

from __future__ import annotations

from pathlib import Path

# Raiz do projeto = dois níveis acima deste arquivo (src/series_eco/config.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Códigos das séries no SGS/BCB.
# IPCA: série principal (variação mensal, % a.m.).
# Câmbio e Selic: variáveis exógenas para os capítulos multivariados.
SGS_SERIES: dict[str, int] = {
    "ipca": 433,      # IPCA - variação mensal (% a.m.)
    "cambio": 1,      # Dólar (venda) - cotação diária
    "selic": 4189,    # Taxa Selic - meta anualizada (% a.a.)
}

# Frequência-alvo do projeto. O IPCA é mensal; séries diárias (câmbio) são
# reamostradas para fechamento mensal nos passos de alinhamento.
TARGET_FREQ: str = "MS"  # month start
