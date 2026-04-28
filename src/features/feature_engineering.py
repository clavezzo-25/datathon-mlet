"""Feature engineering completo baseado no EDA.

Entrada:
    data/raw/stock_data.csv

Saídas principais:
    data/raw/stock_features.csv
    models/scaler_features.joblib
    models/pca_features.joblib

Relatórios:
    reports/correlation_matrix.csv
    reports/removed_by_correlation.csv
    reports/vif_report.csv
    reports/selected_features.csv
    reports/pca_explained_variance.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


INPUT_PATH = Path("data/raw/stock_data.csv")
OUTPUT_PATH = Path("data/raw/stock_features.csv")

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

SCALER_PATH = MODELS_DIR / "scaler_features.joblib"
PCA_PATH = MODELS_DIR / "pca_features.joblib"

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Features exatamente alinhadas ao EDA
EDA_FEATURES = [
    "retorno",
    "volatilidade_20",
    "mm_20",
    "mm_50",
    "amplitude",
    "variacao_dia",
]

TARGET_COLUMN = "target_next_close"

CORRELATION_THRESHOLD = 0.90
VIF_THRESHOLD = 10.0
PCA_VARIANCE_TARGET = 0.95


def load_data() -> pd.DataFrame:
    """Carrega o CSV bruto gerado pelo yfinance."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_PATH}")

    try:
        df = pd.read_csv(INPUT_PATH, header=[0, 1], index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)

    df.index.name = "Date"
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Valida colunas mínimas vindas do yfinance."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    if df.empty:
        raise ValueError("DataFrame vazio recebido.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza básica dos dados brutos."""
    validate_schema(df)

    dados = df.copy()

    for col in REQUIRED_COLUMNS:
        dados[col] = pd.to_numeric(dados[col], errors="coerce")

    dados = dados.sort_index()
    dados = dados[~dados.index.duplicated(keep="first")]
    dados = dados.replace([np.inf, -np.inf], np.nan)

    dados = dados.dropna(subset=REQUIRED_COLUMNS)

    dados = dados[
        (dados["Open"] > 0)
        & (dados["High"] > 0)
        & (dados["Low"] > 0)
        & (dados["Close"] > 0)
        & (dados["Volume"] >= 0)
    ]

    return dados


def compute_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria somente as features definidas no EDA."""
    dados = clean_data(df)

    dados["retorno"] = dados["Close"].pct_change()
    dados["volatilidade_20"] = dados["retorno"].rolling(20).std()
    dados["mm_20"] = dados["Close"].rolling(20).mean()
    dados["mm_50"] = dados["Close"].rolling(50).mean()
    dados["amplitude"] = dados["High"] - dados["Low"]
    dados["variacao_dia"] = dados["Close"] - dados["Open"]

    # Target para seleção automática de features
    dados[TARGET_COLUMN] = dados["Close"].shift(-1)

    dados = dados.replace([np.inf, -np.inf], np.nan)
    dados = dados.dropna().copy()

    return dados


def treat_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Winsorização simples por quantis 1% e 99%."""
    dados = df.copy()

    for col in columns:
        if col not in dados.columns:
            continue

        lower = dados[col].quantile(0.01)
        upper = dados[col].quantile(0.99)
        dados[col] = dados[col].clip(lower=lower, upper=upper)

    return dados


def save_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> None:
    """Salva matriz de correlação em reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    corr = df[columns].corr()
    corr.to_csv(REPORTS_DIR / "correlation_matrix.csv")


def remove_high_correlation(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove features altamente correlacionadas."""
    dados = df.copy()

    corr_matrix = dados[columns].corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    removed_cols = [
        col for col in upper.columns if any(upper[col] > threshold)
    ]

    if removed_cols:
        dados = dados.drop(columns=removed_cols)

    pd.DataFrame(
        {
            "removed_feature": removed_cols,
            "reason": f"correlation_above_{threshold}",
        }
    ).to_csv(REPORTS_DIR / "removed_by_correlation.csv", index=False)

    return dados, removed_cols


def calculate_vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calcula VIF para as features."""
    X = df[columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    vif_data = []

    for i, col in enumerate(X.columns):
        try:
            vif_value = variance_inflation_factor(X.values, i)
        except Exception:
            vif_value = np.nan

        vif_data.append(
            {
                "feature": col,
                "vif": vif_value,
            }
        )

    vif_df = pd.DataFrame(vif_data)
    vif_df["high_vif"] = vif_df["vif"] > VIF_THRESHOLD

    return vif_df.sort_values("vif", ascending=False)


def remove_high_vif(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = VIF_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove iterativamente features com VIF alto."""
    dados = df.copy()
    remaining = columns.copy()
    removed = []

    while len(remaining) > 1:
        vif_df = calculate_vif(dados, remaining)
        max_vif_row = vif_df.iloc[0]
        max_vif = max_vif_row["vif"]
        feature = max_vif_row["feature"]

        if pd.isna(max_vif) or max_vif <= threshold:
            break

        remaining.remove(feature)
        removed.append(feature)

    final_vif = calculate_vif(dados, remaining)
    final_vif.to_csv(REPORTS_DIR / "vif_report.csv", index=False)

    if removed:
        dados = dados.drop(columns=removed)

    pd.DataFrame(
        {
            "removed_feature": removed,
            "reason": f"vif_above_{threshold}",
        }
    ).to_csv(REPORTS_DIR / "removed_by_vif.csv", index=False)

    return dados, removed


def automatic_feature_selection(
    df: pd.DataFrame,
    columns: list[str],
    target_col: str = TARGET_COLUMN,
) -> list[str]:
    """Seleciona automaticamente as melhores features usando SelectKBest."""
    X = df[columns].copy()
    y = df[target_col].copy()

    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    if len(columns) <= 2:
        selected = columns
    else:
        k = max(2, int(np.ceil(len(columns) * 0.70)))
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)

        selected = list(X.columns[selector.get_support()])

        scores = pd.DataFrame(
            {
                "feature": X.columns,
                "score": selector.scores_,
                "selected": selector.get_support(),
            }
        ).sort_values("score", ascending=False)

        scores.to_csv(REPORTS_DIR / "feature_selection_scores.csv", index=False)

    pd.DataFrame({"selected_feature": selected}).to_csv(
        REPORTS_DIR / "selected_features.csv",
        index=False,
    )

    return selected


def scale_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Aplica RobustScaler nas features selecionadas."""
    dados = df.copy()

    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(dados[columns])

    scaled_columns = [f"{col}_scaled" for col in columns]

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=scaled_columns,
        index=dados.index,
    )

    dados = pd.concat([dados, scaled_df], axis=1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "features": columns,
            "scaled_features": scaled_columns,
        },
        SCALER_PATH,
    )

    return dados, scaled_columns


def apply_pca(
    df: pd.DataFrame,
    scaled_columns: list[str],
    variance_target: float = PCA_VARIANCE_TARGET,
) -> pd.DataFrame:
    """Aplica PCA nas features escaladas."""
    dados = df.copy()

    X = dados[scaled_columns]

    pca_full = PCA()
    pca_full.fit(X)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative_variance, variance_target) + 1)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)

    pca_columns = [f"pca_{i + 1}" for i in range(n_components)]

    pca_df = pd.DataFrame(
        components,
        columns=pca_columns,
        index=dados.index,
    )

    dados = pd.concat([dados, pca_df], axis=1)

    joblib.dump(
        {
            "pca": pca,
            "input_features": scaled_columns,
            "pca_features": pca_columns,
            "variance_target": variance_target,
        },
        PCA_PATH,
    )

    pca_report = pd.DataFrame(
        {
            "component": pca_columns,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    pca_report.to_csv(REPORTS_DIR / "pca_explained_variance.csv", index=False)

    return dados


def run_pipeline() -> pd.DataFrame:
    """Executa pipeline completo e adaptável."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = compute_eda_features(df)

    available_features = [col for col in EDA_FEATURES if col in df.columns]

    df = treat_outliers(df, available_features)

    save_correlation_matrix(df, available_features)

    df, removed_corr = remove_high_correlation(df, available_features)
    features_after_corr = [
        col for col in available_features if col not in removed_corr
    ]

    df, removed_vif = remove_high_vif(df, features_after_corr)
    features_after_vif = [
        col for col in features_after_corr if col not in removed_vif
    ]

    selected_features = automatic_feature_selection(df, features_after_vif)

    final_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        TARGET_COLUMN,
        *EDA_FEATURES,
    ]

    # Mantém somente colunas que realmente existem após remoções
    final_columns = [col for col in final_columns if col in df.columns]

    df = df[final_columns].copy()

    df, scaled_columns = scale_features(df, selected_features)

    df = apply_pca(df, scaled_columns)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()

    return df


def save_features(df: pd.DataFrame) -> None:
    """Salva CSV tratado."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH)


def main() -> None:
    print("Iniciando feature engineering...")

    df = run_pipeline()
    save_features(df)

    print(f"Features salvas em: {OUTPUT_PATH}")
    print(f"Scaler salvo em: {SCALER_PATH}")
    print(f"PCA salvo em: {PCA_PATH}")
    print(f"Relatórios salvos em: {REPORTS_DIR}")
    print(f"Linhas finais: {len(df)}")
    print(f"Colunas finais: {len(df.columns)}")


if __name__ == "__main__":
    main()