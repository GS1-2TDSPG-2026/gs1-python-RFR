import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURES = ["ph", "temperatura", "turbidez", "luminosidade", "radiacaoPar"]
JANELA = 7

SCALER_PATH = os.path.join(os.path.dirname(__file__), "../../artifacts/scaler.pkl")


def preparar_sequencia(historico: list[dict]) -> np.ndarray:
    df = pd.DataFrame([{col: d[col] for col in FEATURES} for d in historico])

    if os.path.exists(SCALER_PATH):
        scaler: StandardScaler = joblib.load(SCALER_PATH)
        scaled = scaler.transform(df)
    else:
        scaled = df.values

    return scaled.reshape(1, len(historico), len(FEATURES))


def criar_janelas(df: pd.DataFrame, janela: int = JANELA):
    X, y = [], []
    for i in range(janela, len(df)):
        X.append(df[FEATURES].iloc[i - janela:i].values)
        y.append(df["biomassa"].iloc[i])
    return np.array(X), np.array(y)


def calcular_status(biomassa: float) -> str:
    if biomassa >= 15.0:
        return "CRESCIMENTO_IDEAL"
    elif biomassa >= 10.0:
        return "CRESCIMENTO_MODERADO"
    elif biomassa >= 5.0:
        return "CRESCIMENTO_LENTO"
    else:
        return "ALERTA_BAIXA_BIOMASSA"
