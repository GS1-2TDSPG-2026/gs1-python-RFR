import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = ["ph", "temperatura", "turbidez", "luminosidade", "radiacaoPar"]
TARGET = "biomassa"

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")


def carregar_dados_csv(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    print(f"CSV carregado: {len(df)} registros")
    return df


def treinar(df: pd.DataFrame) -> None:
    X = df[FEATURES].values
    y = df[TARGET].values

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled  = scaler.transform(X_teste)

    print("Treinando Random Forest...")
    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_treino_scaled, y_treino)

    y_pred = modelo.predict(X_teste_scaled)
    mae = mean_absolute_error(y_teste, y_pred)
    r2  = r2_score(y_teste, y_pred)

    print(f"\nMAE: {mae:.3f} g/L | R²: {r2:.3f}")

    importancias = pd.Series(modelo.feature_importances_, index=FEATURES)
    print("\nImportância das variáveis:")
    for feat, imp in importancias.sort_values(ascending=False).items():
        print(f"  {feat}: {imp:.3f}")

    Path(ARTIFACTS_DIR).mkdir(exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("[TRAIN] Modelo salvo")


def treinar_modelo() -> None:
    """
    Wrapper público chamado por main.py e scheduler.py.
    Prioridade: Oracle > CSV histórico > sintético.
    """
    csv_path = os.path.join(ARTIFACTS_DIR, "dataset_historico.csv")

    df = None

    if os.getenv("ORACLE_USER"):
        try:
            from app.db.oracle import carregar_dados_treino
            df = carregar_dados_treino()
            print("Usando dados reais do Oracle")
        except Exception as e:
            print(f"Oracle falhou ({e}), tentando CSV...")

    if df is None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Usando dataset histórico: {len(df)} registros")

    if df is None:
        print("Gerando dados sintéticos...")
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "ph":           np.random.uniform(6.0, 9.0, n),
            "temperatura":  np.random.uniform(18.0, 35.0, n),
            "turbidez":     np.random.uniform(10.0, 80.0, n),
            "luminosidade": np.random.uniform(400.0, 1200.0, n),
            "radiacaoPar":  np.random.uniform(3.0, 9.0, n),
        })
        df["biomassa"] = (
            2.0 * df["radiacaoPar"]
            + 0.3 * df["luminosidade"] / 100
            - 0.5 * (df["ph"] - 7.5).abs()
            - 0.1 * df["turbidez"]
            + 0.2 * df["temperatura"]
            + np.random.normal(0, 0.5, n)
        ).clip(1.0, 25.0)

    treinar(df)


if __name__ == "__main__":
    treinar_modelo()
    print("\nTreinamento concluído!")
