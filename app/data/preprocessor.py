import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURES = ["ph", "temperatura", "turbidez", "luminosidade", "radiacaoPar"]
JANELA = 7  # mínimo de leituras exigido pelo schema

SCALER_PATH = os.path.join(os.path.dirname(__file__), "../../artifacts/scaler.pkl")


def preparar_entrada(dados: dict) -> np.ndarray:
    """Para o Random Forest — usa uma única leitura."""
    df = pd.DataFrame([{col: dados[col] for col in FEATURES}])
    if os.path.exists(SCALER_PATH):
        scaler: StandardScaler = joblib.load(SCALER_PATH)
        return scaler.transform(df)
    return df.values


def preparar_sequencia(historico: list[dict]) -> np.ndarray:
    """Para o LSTM — recebe lista de leituras e retorna shape (1, janela, features)."""
    df = pd.DataFrame([{col: d[col] for col in FEATURES} for d in historico])
    if os.path.exists(SCALER_PATH):
        scaler: StandardScaler = joblib.load(SCALER_PATH)
        df[FEATURES] = scaler.transform(df[FEATURES])
    # Usa as últimas JANELA leituras
    janela = df[FEATURES].values[-JANELA:]
    return janela.reshape(1, JANELA, len(FEATURES))


def criar_janelas(df: pd.DataFrame, janela: int):
    """Cria X e y para treino do LSTM com janela deslizante."""
    X, y = [], []
    valores = df[FEATURES].values
    targets = df["biomassa"].values
    for i in range(len(df) - janela):
        X.append(valores[i:i + janela])
        y.append(targets[i + janela])
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
