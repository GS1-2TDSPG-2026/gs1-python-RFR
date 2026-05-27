"""
predictor.py
------------
Coração da IA. Carrega o modelo treinado (.pkl) e faz a previsão.

Como funciona:
1. O modelo é carregado UMA vez quando a API sobe (não a cada requisição — isso seria lento).
2. Para cada requisição, os dados são preparados e passados para o modelo.
3. O modelo retorna um número (biomassa estimada).
4. Calculamos a data de colheita com base numa taxa média de crescimento.
"""

import os
import joblib
import numpy as np
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor

from app.data.preprocessor import preparar_entrada, calcular_status
from app.models.schemas import EntradaPrevisao, SaidaPrevisao

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../artifacts/model.pkl")

# Variável global: modelo carregado na memória (carrega 1x ao iniciar a API)
_modelo: RandomForestRegressor | None = None


def carregar_modelo() -> RandomForestRegressor:
    """
    Carrega o modelo do disco. Se não existir, cria um modelo básico de demonstração.
    Em produção, o modelo SEMPRE deve existir (gerado pelo train.py).
    """
    global _modelo

    if _modelo is not None:
        return _modelo  # Já está na memória, não carrega de novo

    if os.path.exists(MODEL_PATH):
        _modelo = joblib.load(MODEL_PATH)
        print(f"✅ Modelo carregado de: {MODEL_PATH}")
    else:
        # Modelo de demonstração: treinado com dados sintéticos
        # Usado apenas para testar a API sem ter dados reais ainda
        print("⚠️  Modelo não encontrado. Usando modelo de demonstração.")
        _modelo = _criar_modelo_demo()

    return _modelo


def _criar_modelo_demo() -> RandomForestRegressor:
    """
    Cria e treina um modelo simples com dados fictícios.
    Só serve para demonstrar que a API funciona.
    Em produção, substitua pelo train.py com dados reais do Oracle.
    """
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import joblib

    # Dados sintéticos que simulam condições de crescimento de microalgas
    dados_treino = [
        # ph,  temp,  turb,  lumi,  par,  biomassa
        [7.0, 25.0, 30.0, 800.0, 5.0,  12.0],
        [7.5, 27.0, 35.0, 900.0, 6.0,  15.5],
        [6.5, 22.0, 45.0, 600.0, 4.0,   8.0],
        [8.0, 30.0, 20.0, 1000.0, 7.0, 18.0],
        [7.2, 26.0, 38.0, 820.0, 5.7,  14.8],
        [6.8, 23.0, 50.0, 700.0, 4.5,   9.5],
        [7.8, 28.0, 25.0, 950.0, 6.5,  17.0],
        [7.1, 24.0, 40.0, 750.0, 5.2,  11.0],
        [6.0, 20.0, 60.0, 500.0, 3.5,   5.0],
        [8.5, 32.0, 15.0, 1100.0, 8.0, 20.0],
    ]

    import numpy as np
    dados = np.array(dados_treino)
    X = dados[:, :-1]   # Features (tudo menos a última coluna)
    y = dados[:, -1]    # Target (última coluna = biomassa)

    # Normaliza os dados e salva o scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(scaler, MODEL_PATH.replace("model.pkl", "scaler.pkl"))

    # Treina o Random Forest
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_scaled, y)
    joblib.dump(modelo, MODEL_PATH)

    return modelo


def prever(entrada: EntradaPrevisao) -> SaidaPrevisao:
    """
    Função principal: recebe os dados do tanque e retorna a previsão.

    Parâmetros:
        entrada: objeto validado pelo Pydantic com todos os sensores

    Retorna:
        SaidaPrevisao: biomassa estimada, data de colheita, confiança e status
    """
    modelo = carregar_modelo()

    # 1. Prepara os dados (normalização)
    X = preparar_entrada(entrada.model_dump())

    # 2. Faz a previsão
    biomassa_estimada: float = round(float(modelo.predict(X)[0]), 2)

    # 3. Estima a data de colheita
    # Lógica: biomassa cresce ~0.5g/L por dia em condições normais
    # Colheita quando atingir 20g/L (limiar de colheita ideal)
    LIMIAR_COLHEITA = 20.0
    TAXA_CRESCIMENTO_DIARIA = 0.5  # g/L por dia (ajustar com dados reais)

    if biomassa_estimada >= LIMIAR_COLHEITA:
        dias_para_colheita = 0
    else:
        dias_para_colheita = int((LIMIAR_COLHEITA - biomassa_estimada) / TAXA_CRESCIMENTO_DIARIA)

    data_colheita = date.today() + timedelta(days=dias_para_colheita)

    # 4. Calcula confiança baseada na variância das árvores do Random Forest
    # Cada árvore da floresta dá um valor. Quanto mais parecidas, maior a confiança.
    predicoes_arvores = np.array([arvore.predict(X)[0] for arvore in modelo.estimators_])
    desvio = float(np.std(predicoes_arvores))
    confianca = round(max(0.0, min(100.0, 100.0 - (desvio * 10))), 1)

    # 5. Define o status
    status = calcular_status(biomassa_estimada)

    return SaidaPrevisao(
        tanqueId=entrada.tanqueId,
        biomassaEstimada=biomassa_estimada,
        dataPrevistaColheita=data_colheita,
        confianca=confianca,
        status=status,
    )
