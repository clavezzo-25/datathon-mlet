import argparse
import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM

from src.features import feature_engineering as fe


ROOT_DIR = Path(__file__).resolve().parents[2]
MLFLOW_DIR = ROOT_DIR / "mlflow"
MLFLOW_DIR.mkdir(exist_ok=True)
(MLFLOW_DIR / "artifacts").mkdir(exist_ok=True)

RAW_DATA_PATH = Path("data/raw/stock_data.csv")
FEATURES_DATA_PATH = Path("data/raw/stock_features.csv")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MLP_PyTorch(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def ensure_features_dataset(data_path: str | None = None) -> str:
    """
    Garante que o arquivo data/raw/stock_features.csv exista.

    Se o arquivo não existir, gera automaticamente usando feature_engineering.py.
    """
    if data_path:
        return data_path

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo bruto não encontrado: {RAW_DATA_PATH}. "
            "Execute primeiro o ingest.py."
        )

    should_generate = not FEATURES_DATA_PATH.exists()

    if FEATURES_DATA_PATH.exists():
        raw_mtime = RAW_DATA_PATH.stat().st_mtime
        features_mtime = FEATURES_DATA_PATH.stat().st_mtime
        should_generate = raw_mtime > features_mtime

    if should_generate:
        logger.info("Gerando features automaticamente...")
        features_df = fe.run_pipeline()
        fe.save_features(features_df)
        logger.info("Features geradas em %s", FEATURES_DATA_PATH)

    return str(FEATURES_DATA_PATH)


def get_feature_columns(dados: pd.DataFrame) -> list[str]:
    """Seleciona automaticamente features tratadas."""
    scaled_cols = [col for col in dados.columns if col.endswith("_scaled")]
    pca_cols = [col for col in dados.columns if col.startswith("pca_")]

    feature_cols = scaled_cols + pca_cols

    if not feature_cols:
        raise ValueError(
            "Nenhuma feature tratada encontrada. "
            "Execute: python src/features/feature_engineering.py"
        )

    return feature_cols


def carregar_dados_csv(csv_path: str) -> pd.DataFrame:
    logger.info("Lendo dados do CSV: %s", csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    try:
        dados = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
    except Exception:
        dados = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if "Close" not in dados.columns:
        raise ValueError(
            f"A coluna 'Close' não foi encontrada no CSV. "
            f"Colunas encontradas: {list(dados.columns)}"
        )

    if "target_next_close" not in dados.columns:
        raise ValueError(
            "A coluna 'target_next_close' não foi encontrada. "
            "Execute o feature_engineering.py antes do treino."
        )

    dados["Close"] = pd.to_numeric(dados["Close"], errors="coerce")
    dados["target_next_close"] = pd.to_numeric(dados["target_next_close"], errors="coerce")
    dados = dados.dropna(subset=["Close", "target_next_close"])

    return dados


def preparar_series(precos: np.ndarray, janela_dias: int):
    """
    Função legada mantida para compatibilidade com testes antigos.
    Usa apenas uma série de preços.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    precos_normalizados = scaler.fit_transform(precos)

    X, y = [], []

    for i in range(janela_dias, len(precos_normalizados)):
        X.append(precos_normalizados[i - janela_dias:i, 0])
        y.append(precos_normalizados[i, 0])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("Dados insuficientes para a janela escolhida.")

    return X, y, scaler


def preparar_series_features(
    dados: pd.DataFrame,
    janela_dias: int,
    target_col: str = "target_next_close",
):
    """
    Prepara série temporal multivariada com features tratadas.

    Retorna:
        X_3d: usado no Keras/LSTM
        y: target escalado
        scaler_y: scaler do target
        feature_cols: features utilizadas
    """
    feature_cols = get_feature_columns(dados)

    X_raw = dados[feature_cols].values
    y_raw = dados[[target_col]].values

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y_raw)

    X, y = [], []

    for i in range(janela_dias, len(dados)):
        X.append(X_raw[i - janela_dias:i])
        y.append(y_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("Dados insuficientes para a janela escolhida.")

    return X, y, scaler_y, feature_cols


def treinar_pytorch(X_train: np.ndarray, y_train: np.ndarray, modelo_path: str) -> nn.Module:
    input_dim = X_train.shape[1]
    model = MLP_PyTorch(input_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    for _ in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), modelo_path)
    logger.info("Modelo PyTorch salvo em %s", modelo_path)

    return model


def avaliar_regressao(y_true: np.ndarray, y_pred: np.ndarray, scaler: MinMaxScaler) -> dict:
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / y_true_inv)) * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }


def criar_baseline_naive(
    dados: pd.DataFrame,
    janela_dias: int,
    tamanho_treino: int,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Baseline naive para o conjunto de teste:
    previsão do próximo fechamento = fechamento anterior.
    """
    close_values = dados["Close"].values

    baseline_values = []

    inicio_teste = janela_dias + tamanho_treino

    for i in range(inicio_teste, len(dados)):
        baseline_values.append(close_values[i - 1])

    baseline_values = np.array(baseline_values).reshape(-1, 1)

    return scaler.transform(baseline_values)


def log_tags_padronizadas(model_type: str, framework: str):
    mlflow.set_tag("model_type", model_type)
    mlflow.set_tag("framework", framework)
    mlflow.set_tag("owner", "grupo-XX")
    mlflow.set_tag("phase", "datathon-fase05")
    mlflow.set_tag("problem_type", "regression")
    mlflow.set_tag("dataset_type", "time_series")


def main(args):
    ticker = args.ticker
    start_date = args.start
    end_date = args.end
    janela_dias = args.janela
    epocas = args.epochs
    batchsize = args.batch

    logger.info(
        "Treinando modelos para %s | início=%s | fim=%s | janela=%s",
        ticker,
        start_date,
        end_date,
        janela_dias,
    )

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    data_path = ensure_features_dataset(args.data_path)
    dados = carregar_dados_csv(data_path)

    X_3d, y, scaler, feature_cols = preparar_series_features(dados, janela_dias)

    logger.info("Features utilizadas: %s", feature_cols)
    logger.info("Shape X 3D: %s", X_3d.shape)

    X_flat = X_3d.reshape(X_3d.shape[0], X_3d.shape[1] * X_3d.shape[2])
    X_keras = X_3d

    tamanho_treino = int(len(X_flat) * 0.8)

    X_train, X_test = X_flat[:tamanho_treino], X_flat[tamanho_treino:]
    y_train, y_test = y[:tamanho_treino], y[tamanho_treino:]

    X_train_keras = X_keras[:tamanho_treino]
    X_test_keras = X_keras[tamanho_treino:]

    tracking_uri = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("previsao_acoes")

    with mlflow.start_run(run_name=f"train_{ticker}") as run:
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)
        mlflow.log_param("janela", janela_dias)
        mlflow.log_param("epochs", epocas)
        mlflow.log_param("batch_size", batchsize)
        mlflow.log_param("n_features_original", len(feature_cols))
        mlflow.log_param("n_features_model_input", X_train.shape[1])
        mlflow.log_param("feature_columns", ",".join(feature_cols))
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_test", X_test.shape[0])
        mlflow.log_param("split_type", "temporal_80_20")
        mlflow.log_param("data_source", data_path)

        log_tags_padronizadas(model_type="regression", framework="multiple")

        y_pred_baseline = criar_baseline_naive(
            dados=dados,
            janela_dias=janela_dias,
            tamanho_treino=tamanho_treino,
            scaler=scaler,
        )

        metrics_baseline = avaliar_regressao(y_test, y_pred_baseline, scaler)

        mlflow.log_metric("mae_baseline", metrics_baseline["mae"])
        mlflow.log_metric("rmse_baseline", metrics_baseline["rmse"])
        mlflow.log_metric("mape_baseline", metrics_baseline["mape"])

        logger.info(
            "[Baseline] MAE=%.4f | RMSE=%.4f | MAPE=%.2f",
            metrics_baseline["mae"],
            metrics_baseline["rmse"],
            metrics_baseline["mape"],
        )

        modelo_pytorch_path = os.path.join(models_dir, f"modelo_{ticker}_pytorch.pth")
        model_pytorch = treinar_pytorch(X_train, y_train, modelo_pytorch_path)
        model_pytorch.eval()

        with torch.no_grad():
            y_pred_torch = model_pytorch(torch.tensor(X_test, dtype=torch.float32)).numpy()

        metrics_torch = avaliar_regressao(y_test, y_pred_torch, scaler)

        mlflow.log_metric("mae_pytorch", metrics_torch["mae"])
        mlflow.log_metric("rmse_pytorch", metrics_torch["rmse"])
        mlflow.log_metric("mape_pytorch", metrics_torch["mape"])
        mlflow.pytorch.log_model(model_pytorch, "pytorch_model")

        logger.info(
            "[PyTorch] MAE=%.4f | RMSE=%.4f | MAPE=%.2f",
            metrics_torch["mae"],
            metrics_torch["rmse"],
            metrics_torch["mape"],
        )

        modelo_sklearn_path = os.path.join(models_dir, f"modelo_{ticker}_sklearn.joblib")
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)

        joblib.dump(mlp, modelo_sklearn_path)

        y_pred_sklearn = mlp.predict(X_test).reshape(-1, 1)
        metrics_sklearn = avaliar_regressao(y_test, y_pred_sklearn, scaler)

        mlflow.log_metric("mae_sklearn", metrics_sklearn["mae"])
        mlflow.log_metric("rmse_sklearn", metrics_sklearn["rmse"])
        mlflow.log_metric("mape_sklearn", metrics_sklearn["mape"])
        mlflow.sklearn.log_model(mlp, "sklearn_model")

        logger.info(
            "[Scikit] MAE=%.4f | RMSE=%.4f | MAPE=%.2f",
            metrics_sklearn["mae"],
            metrics_sklearn["rmse"],
            metrics_sklearn["mape"],
        )

        if args.keras:
            logger.info("Treinando Keras...")

            modelo = Sequential(
                [
                    Input(shape=(X_train_keras.shape[1], X_train_keras.shape[2])),
                    LSTM(units=50, return_sequences=True),
                    Dropout(0.2),
                    LSTM(units=50, return_sequences=False),
                    Dropout(0.2),
                    Dense(units=1),
                ]
            )

            modelo.compile(optimizer="adam", loss="mean_squared_error")

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
            )

            modelo.fit(
                X_train_keras,
                y_train,
                epochs=epocas,
                batch_size=batchsize,
                validation_data=(X_test_keras, y_test),
                callbacks=[early_stop],
                verbose=1,
            )

            modelo_keras_path = os.path.join(models_dir, f"modelo_{ticker}.keras")
            modelo.save(modelo_keras_path)

            y_pred_keras = modelo.predict(X_test_keras)
            metrics_keras = avaliar_regressao(y_test, y_pred_keras, scaler)

            mlflow.log_metric("mae_keras", metrics_keras["mae"])
            mlflow.log_metric("rmse_keras", metrics_keras["rmse"])
            mlflow.log_metric("mape_keras", metrics_keras["mape"])
            mlflow.tensorflow.log_model(modelo, "keras_model")

            logger.info(
                "[Keras] MAE=%.4f | RMSE=%.4f | MAPE=%.2f",
                metrics_keras["mae"],
                metrics_keras["rmse"],
                metrics_keras["mape"],
            )

        logger.info("Run finalizado com run_id=%s", run.info.run_id)
        print(f"run_id={run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelos para previsão de preços")

    parser.add_argument("--ticker", type=str, default="ITUB4.SA", help="Código do ativo")
    parser.add_argument("--start", type=str, default="2025-04-01", help="Data inicial")
    parser.add_argument("--end", type=str, default="2027-04-30", help="Data final")
    parser.add_argument("--janela", type=int, default=90, help="Tamanho da janela de dias")
    parser.add_argument("--epochs", type=int, default=40, help="Número de épocas")
    parser.add_argument("--batch", type=int, default=32, help="Tamanho do batch")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping do Keras")
    parser.add_argument("--keras", action="store_true", help="Treinar também modelo Keras")

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Opcional. Se não informado, usa automaticamente data/raw/stock_features.csv",
    )

    args = parser.parse_args()
    main(args)