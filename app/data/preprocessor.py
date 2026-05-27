"""
preprocessor.py
---------------
Prepara os dados brutos antes de entrar no modelo de IA.
O modelo foi treinado com dados numa escala específica (ex: temperatura entre 0-50).
Se você jogar um número fora dessa escala sem normalizar, a previsão fica errada.
O StandardScaler resolve isso: transforma tudo para a mesma escala.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ordem exata das colunas que o modelo espera receber
# IMPORTANTE: nunca mudar a ordem sem retreinar o modelo
FEATURES = ["ph", "temperatura", "turbidez", "luminosidade", "radiacaoPar"]

SCALER_PATH = os.path.join(os.path.dirname(__file__), "../../artifacts/scaler.pkl")


def preparar_entrada(dados: dict) -> np.ndarray:
    """
    Recebe o dicionário de entrada e retorna um array numpy pronto para o modelo.

    Exemplo:
        entrada = {"ph": 7.2, "temperatura": 26.5, ...}
        saída   = array([[0.12, -0.34, 0.78, 1.02, -0.55]])  ← valores normalizados
    """
    # Monta um DataFrame com UMA linha na ordem certa das features
    df = pd.DataFrame([{col: dados[col] for col in FEATURES}])

    # Se o scaler já foi salvo (depois do treino), usa ele
    if os.path.exists(SCALER_PATH):
        scaler: StandardScaler = joblib.load(SCALER_PATH)
        return scaler.transform(df)

    # Se não existe scaler (ambiente de dev / primeiro uso), retorna os valores crus
    return df.values


def calcular_status(biomassa: float) -> str:
    """
    Classifica o status do crescimento com base na biomassa estimada.
    Esses limiares podem ser ajustados conforme o conhecimento do domínio.
    """
    if biomassa >= 15.0:
        return "CRESCIMENTO_IDEAL"
    elif biomassa >= 10.0:
        return "CRESCIMENTO_MODERADO"
    elif biomassa >= 5.0:
        return "CRESCIMENTO_LENTO"
    else:
        return "ALERTA_BAIXA_BIOMASSA"
