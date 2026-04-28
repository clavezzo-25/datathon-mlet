import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.models.baseline import (
    carregar_dados_csv,
    avaliar_modelo,
    treinar_pytorch,
    carregar_pytorch,
    preparar_series_features,
    get_feature_columns,
    criar_baseline_naive,
    ensure_features_dataset,
)


@pytest.fixture
def sample_stock_csv(tmp_path):
    csv_path = tmp_path / "stock.csv"

    df = pd.DataFrame(
        {
            "Close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500],
            "return_1d_scaled": [0.01, 0.02, 0.01, 0.03, 0.02, 0.01],
            "volatility_20_scaled": [0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
            "pca_1": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "pca_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "target_next_close": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
    )

    df.to_csv(csv_path)
    return csv_path


def test_carregar_dados_ok(sample_stock_csv):
    df = carregar_dados_csv(str(sample_stock_csv))

    assert not df.empty
    assert "Close" in df.columns
    assert "target_next_close" in df.columns
    assert "pca_1" in df.columns
    assert "pca_2" in df.columns


def test_carregar_dados_sem_close(tmp_path):
    csv_path = tmp_path / "stock_sem_close.csv"

    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "target_next_close": [11.0, 12.0, 13.0],
            "pca_1": [0.1, 0.2, 0.3],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="D"),
    )

    df.to_csv(csv_path)

    with pytest.raises(ValueError, match="Close"):
        carregar_dados_csv(str(csv_path))


def test_get_feature_columns(sample_stock_csv):
    df = carregar_dados_csv(str(sample_stock_csv))

    feature_cols = get_feature_columns(df)

    assert "return_1d_scaled" in feature_cols
    assert "volatility_20_scaled" in feature_cols
    assert "pca_1" in feature_cols
    assert "pca_2" in feature_cols


def test_preparar_series_features(sample_stock_csv):
    df = carregar_dados_csv(str(sample_stock_csv))

    X, y, scaler, feature_cols = preparar_series_features(df, janela_dias=2)

    assert X.shape[0] > 0
    assert X.shape[1] == 2
    assert X.shape[2] == len(feature_cols)
    assert len(y) == X.shape[0]
    assert scaler is not None
    assert len(feature_cols) > 0


def test_avaliar_modelo():
    y_real_original = np.array([[10.0], [20.0], [30.0]])
    y_pred_original = np.array([[11.0], [19.0], [29.0]])

    scaler = MinMaxScaler()
    scaler.fit(y_real_original)

    y_real = scaler.transform(y_real_original).reshape(-1)
    y_pred = scaler.transform(y_pred_original).reshape(-1)

    metricas = avaliar_modelo(
        y_real,
        y_pred,
        "modelo_teste",
        scaler,
    )

    assert isinstance(metricas, dict)
    assert "Modelo" in metricas
    assert "MAE" in metricas
    assert "RMSE" in metricas
    assert "MAPE (%)" in metricas
    assert "Predicoes" in metricas
    assert metricas["Modelo"] == "modelo_teste"
    assert metricas["MAE"] >= 0
    assert metricas["RMSE"] >= 0
    assert metricas["MAPE (%)"] >= 0


def test_pytorch_train_and_load(tmp_path):
    modelo_path = tmp_path / "modelo_teste_pytorch.pth"

    X = np.array(
        [
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5],
            [6.0, 0.6],
        ],
        dtype=np.float32,
    )

    y = np.array(
        [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
        dtype=np.float32,
    )

    model = treinar_pytorch(X, y, modelo_path=str(modelo_path))

    assert model is not None
    assert modelo_path.exists()

    preds = carregar_pytorch(X, modelo_path=str(modelo_path))

    assert preds.shape[0] == X.shape[0]
    assert np.all(np.isfinite(preds))


def test_criar_baseline_naive(sample_stock_csv):
    df = carregar_dados_csv(str(sample_stock_csv))

    _, _, scaler, _ = preparar_series_features(df, janela_dias=2)

    y_pred = criar_baseline_naive(df, janela=2, scaler=scaler)

    assert y_pred is not None
    assert len(y_pred) == len(df) - 2
    assert np.all(np.isfinite(y_pred))


def test_ensure_features_dataset_com_data_path(tmp_path):
    fake_path = tmp_path / "stock_features.csv"
    fake_path.write_text("dummy")

    result = ensure_features_dataset(str(fake_path))

    assert result == str(fake_path)


def test_get_feature_columns_sem_features_tratadas(sample_stock_csv):
    df = carregar_dados_csv(str(sample_stock_csv))

    df = df[["Close", "Volume", "target_next_close"]]

    with pytest.raises(ValueError, match="Nenhuma feature tratada"):
        get_feature_columns(df)