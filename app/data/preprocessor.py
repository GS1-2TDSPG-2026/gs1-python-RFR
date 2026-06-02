"""
preprocessor.py
---------------
Prepara os dados brutos antes de entrar no modelo Random Forest.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURES = ["ph", "temperatura", "turbidez", "luminosidade", "radiacaoPar"]

SCALER_PATH = os.path.join(os.path.dirname(__file__), "../../artifacts/scaler.pkl")


def preparar_entrada(dados: dict) -> np.ndarray:
    """
    Recebe o dicionário de entrada e retorna array numpy pronto para o modelo.
    Usa a última leitura do histórico (a mais recente).
    """
    df = pd.DataFrame([{col: dados[col] for col in FEATURES}])

    if os.path.exists(SCALER_PATH):
        scaler: StandardScaler = joblib.load(SCALER_PATH)
        return scaler.transform(df)

    return df.values


def calcular_status(biomassa: float) -> str:
    if biomassa >= 15.0:
        return "CRESCIMENTO_IDEAL"
    elif biomassa >= 10.0:
        return "CRESCIMENTO_MODERADO"
    elif biomassa >= 5.0:
        return "CRESCIMENTO_LENTO"
    else:
        return "ALERTA_BAIXA_BIOMASSA"
