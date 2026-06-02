"""
predictor.py
------------
Coração da IA. Carrega o modelo Random Forest e faz a previsão.
"""

import os
import joblib
import numpy as np
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor

from app.data.preprocessor import preparar_entrada, calcular_status
from app.models.schemas import EntradaPrevisao, SaidaPrevisao

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../artifacts/model.pkl")

_modelo: RandomForestRegressor | None = None


def carregar_modelo() -> RandomForestRegressor:
    global _modelo

    if _modelo is not None:
        return _modelo

    if os.path.exists(MODEL_PATH):
        _modelo = joblib.load(MODEL_PATH)
        print(f"Modelo carregado: {MODEL_PATH}")
    else:
        print("Modelo não encontrado. Usando modelo de demonstração.")
        _modelo = _criar_modelo_demo()

    return _modelo


def _criar_modelo_demo() -> RandomForestRegressor:
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    dados_treino = [
        # ph,  temp,  turb,  lumi,   par,  biomassa
        [7.0, 25.0, 30.0,  800.0, 5.0,  12.0],
        [7.5, 27.0, 35.0,  900.0, 6.0,  15.5],
        [6.5, 22.0, 45.0,  600.0, 4.0,   8.0],
        [8.0, 30.0, 20.0, 1000.0, 7.0,  18.0],
        [7.2, 26.0, 38.0,  820.0, 5.7,  14.8],
        [6.8, 23.0, 50.0,  700.0, 4.5,   9.5],
        [7.8, 28.0, 25.0,  950.0, 6.5,  17.0],
        [7.1, 24.0, 40.0,  750.0, 5.2,  11.0],
        [6.0, 20.0, 60.0,  500.0, 3.5,   5.0],
        [8.5, 32.0, 15.0, 1100.0, 8.0,  20.0],
    ]

    dados = np.array(dados_treino)
    X = dados[:, :-1]
    y = dados[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(scaler, MODEL_PATH.replace("model.pkl", "scaler.pkl"))

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_scaled, y)
    joblib.dump(modelo, MODEL_PATH)

    return modelo


def prever(entrada: EntradaPrevisao) -> SaidaPrevisao:
    modelo = carregar_modelo()

    # Usa a leitura mais recente do histórico
    ultima_leitura = entrada.historico[-1].model_dump()
    X = preparar_entrada(ultima_leitura)

    biomassa_estimada: float = round(float(modelo.predict(X)[0]), 2)

    LIMIAR_COLHEITA = 20.0
    TAXA_CRESCIMENTO_DIARIA = 0.5

    dias_para_colheita = (
        0 if biomassa_estimada >= LIMIAR_COLHEITA
        else int((LIMIAR_COLHEITA - biomassa_estimada) / TAXA_CRESCIMENTO_DIARIA)
    )
    data_colheita = date.today() + timedelta(days=dias_para_colheita)

    # Confiança baseada na variância das árvores
    predicoes_arvores = np.array([
        arvore.predict(X)[0] for arvore in modelo.estimators_
    ])
    desvio = float(np.std(predicoes_arvores))
    confianca = round(max(0.0, min(100.0, 100.0 - (desvio * 10))), 1)

    status = calcular_status(biomassa_estimada)

    # Salva no Oracle (silencioso se indisponível)
    try:
        from app.db.oracle import salvar_previsao, buscar_ultimo_dado_orbital
        id_dado_orbital = buscar_ultimo_dado_orbital(entrada.tanqueId)
        if id_dado_orbital is not None:
            salvar_previsao(
                tanque_id=entrada.tanqueId,
                id_dado_orbital=id_dado_orbital,
                biomassa=biomassa_estimada,
                data_pico=data_colheita,
                confianca=confianca,
                modelo="RandomForest_v1",
            )
    except Exception as e:
        print(f"Oracle indisponível: {e}")

    return SaidaPrevisao(
        tanqueId=entrada.tanqueId,
        biomassaEstimada=biomassa_estimada,
        dataPrevistaColheita=data_colheita,
        confianca=confianca,
        status=status,
    )
