"""Testes do pipeline de Feature Engineering."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import feature_engineering as fe


def make_stock_dataframe(periods: int = 80) -> pd.DataFrame:
    """Cria dados sintéticos no padrão yfinance."""
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")

    return pd.DataFrame(
        {
            "Open": np.linspace(10, 20, periods),
            "High": np.linspace(11, 21, periods),
            "Low": np.linspace(9, 19, periods),
            "Close": np.linspace(10.5, 20.5, periods),
            "Volume": np.arange(1000, 1000 + periods),
        },
        index=dates,
    )


def test_compute_eda_features_creates_expected_columns():
    df = make_stock_dataframe()

    result = fe.compute_eda_features(df)

    expected_columns = [
        "retorno",
        "volatilidade_20",
        "mm_20",
        "mm_50",
        "amplitude",
        "variacao_dia",
        "target_next_close",
    ]

    for col in expected_columns:
        assert col in result.columns


def test_compute_eda_features_removes_nulls():
    df = make_stock_dataframe()

    result = fe.compute_eda_features(df)

    assert result.isna().sum().sum() == 0


def test_treat_outliers_keeps_row_count():
    df = make_stock_dataframe()
    df = fe.compute_eda_features(df)

    result = fe.treat_outliers(df, fe.EDA_FEATURES)

    assert len(result) == len(df)


def test_remove_high_correlation_returns_dataframe_and_list(tmp_path, monkeypatch):
    df = make_stock_dataframe()
    df = fe.compute_eda_features(df)

    monkeypatch.setattr(fe, "REPORTS_DIR", tmp_path)

    result, removed = fe.remove_high_correlation(
        df,
        fe.EDA_FEATURES,
        threshold=0.90,
    )

    assert isinstance(result, pd.DataFrame)
    assert isinstance(removed, list)
    assert (tmp_path / "removed_by_correlation.csv").exists()


def test_save_correlation_matrix_creates_report(tmp_path, monkeypatch):
    df = make_stock_dataframe()
    df = fe.compute_eda_features(df)

    monkeypatch.setattr(fe, "REPORTS_DIR", tmp_path)

    fe.save_correlation_matrix(df, fe.EDA_FEATURES)

    assert (tmp_path / "correlation_matrix.csv").exists()


def test_calculate_vif_returns_report_dataframe():
    df = make_stock_dataframe()
    df = fe.compute_eda_features(df)

    vif_df = fe.calculate_vif(df, fe.EDA_FEATURES)

    assert isinstance(vif_df, pd.DataFrame)
    assert "feature" in vif_df.columns
    assert "vif" in vif_df.columns


def test_automatic_feature_selection_returns_list(tmp_path, monkeypatch):
    df = make_stock_dataframe()
    df = fe.compute_eda_features(df)

    monkeypatch.setattr(fe, "REPORTS_DIR", tmp_path)

    selected = fe.automatic_feature_selection(
        df,
        fe.EDA_FEATURES,
        target_col=fe.TARGET_COLUMN,
    )

    assert isinstance(selected, list)
    assert len(selected) > 0
    assert (tmp_path / "selected_features.csv").exists()


def test_scale_features_creates_scaled_columns(tmp_path, monkeypatch):
    df = make_stock_dataframe()
    df = fe.compute_eda_features(df)

    monkeypatch.setattr(fe, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(fe, "SCALER_PATH", tmp_path / "scaler_features.joblib")

    result, scaled_columns = fe.scale_features(df, fe.EDA_FEATURES)

    assert isinstance(result, pd.DataFrame)
    assert len(scaled_columns) == len(fe.EDA_FEATURES)

    for col in scaled_columns:
        assert col in result.columns

    assert (tmp_path / "scaler_features.joblib").exists()


def test_apply_pca_creates_pca_columns(tmp_path, monkeypatch):
    df = make_stock_dataframe()
    df = fe.compute_eda_features(df)

    monkeypatch.setattr(fe, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(fe, "PCA_PATH", tmp_path / "pca_features.joblib")

    df_scaled, scaled_columns = fe.scale_features(df, fe.EDA_FEATURES)
    result = fe.apply_pca(df_scaled, scaled_columns)

    pca_cols = [col for col in result.columns if col.startswith("pca_")]

    assert len(pca_cols) > 0
    assert (tmp_path / "pca_features.joblib").exists()
    assert (tmp_path / "pca_explained_variance.csv").exists()


def test_run_pipeline_creates_final_dataframe(tmp_path, monkeypatch):
    df = make_stock_dataframe()

    input_path = tmp_path / "stock_data.csv"
    output_path = tmp_path / "stock_features.csv"
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"

    df.to_csv(input_path)

    monkeypatch.setattr(fe, "INPUT_PATH", input_path)
    monkeypatch.setattr(fe, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(fe, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(fe, "MODELS_DIR", models_dir)
    monkeypatch.setattr(fe, "SCALER_PATH", models_dir / "scaler_features.joblib")
    monkeypatch.setattr(fe, "PCA_PATH", models_dir / "pca_features.joblib")

    result = fe.run_pipeline()

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert fe.TARGET_COLUMN in result.columns
    assert any(col.endswith("_scaled") for col in result.columns)
    assert any(col.startswith("pca_") for col in result.columns)


def test_save_features_creates_csv(tmp_path, monkeypatch):
    df = make_stock_dataframe()
    df = fe.compute_eda_features(df)

    output_path = tmp_path / "stock_features.csv"
    monkeypatch.setattr(fe, "OUTPUT_PATH", output_path)

    fe.save_features(df)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_missing_required_columns_raises_error():
    df = pd.DataFrame(
        {
            "Open": [10, 11],
            "Close": [10.5, 11.5],
        }
    )

    with pytest.raises(ValueError):
        fe.validate_schema(df)