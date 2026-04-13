"""Testes do módulo de carga de dados.

Cobrem o contrato sem tocar a rede: o download do SGS é substituído por dados
sintéticos, e o cache Parquet é redirecionado para um diretório temporário.
"""

from __future__ import annotations

import pandas as pd
import pytest

from series_eco.data import loaders


@pytest.fixture
def fake_raw_dir(tmp_path, monkeypatch):
    """Redireciona o cache Parquet para um diretório temporário."""
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(loaders, "RAW_DIR", raw)
    return raw


def _synthetic_ipca() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    return pd.Series([0.21, 0.25, 0.07, 0.31, 0.45, 0.26], index=idx, name="ipca")


def test_fetch_series_rejects_unknown_name(fake_raw_dir):
    # Arrange / Act / Assert
    with pytest.raises(KeyError):
        loaders.fetch_series("desconhecida", start="2020-01-01")


def test_fetch_series_downloads_then_caches(fake_raw_dir, monkeypatch):
    # Arrange: o download retorna dados sintéticos e conta as chamadas.
    calls = {"n": 0}

    def fake_download(name, start, end):
        calls["n"] += 1
        return _synthetic_ipca()

    monkeypatch.setattr(loaders, "_download_series", fake_download)

    # Act: primeira chamada baixa; segunda deve usar o cache em disco.
    first = loaders.fetch_series("ipca", start="2020-01-01")
    second = loaders.fetch_series("ipca", start="2020-01-01")

    # Assert
    assert calls["n"] == 1, "segunda chamada deveria ler do cache, não baixar"
    assert (fake_raw_dir / "ipca.parquet").exists()
    # check_freq=False: o round-trip em Parquet não preserva o atributo `freq`
    # do índice. Isso é irrelevante aqui — a frequência é reestabelecida no
    # resample de load_panel; o contrato de fetch_series é "valores por data".
    pd.testing.assert_series_equal(first, second, check_freq=False)


def test_load_panel_aligns_and_drops_incomplete_rows(fake_raw_dir, monkeypatch):
    # Arrange: ipca mensal completo; selic mensal começando um mês depois.
    def fake_download(name, start, end):
        if name == "ipca":
            return _synthetic_ipca()
        idx = pd.date_range("2020-02-01", periods=5, freq="MS")
        return pd.Series(range(5), index=idx, name="selic", dtype=float)

    monkeypatch.setattr(loaders, "_download_series", fake_download)

    # Act
    panel = loaders.load_panel("2020-01-01", names=["ipca", "selic"])

    # Assert: a primeira linha (jan/2020, sem selic) deve sair pelo dropna.
    assert list(panel.columns) == ["ipca", "selic"]
    assert panel.index.min() == pd.Timestamp("2020-02-01")
    assert not panel.isna().any().any()
