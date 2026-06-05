import os
import joblib
import numpy as np
from datetime import date, timedelta

from app.data.preprocessor import calcular_status
from app.models.schemas import EntradaPrevisao, SaidaPrevisao

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../../artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

_modelo = None
_scaler = None

def carregar_modelo():
    global _modelo, _scaler
    if _modelo is not None and _scaler is not None:
        return _modelo, _scaler

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        _modelo = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        print("Artefatos RFR carregados com sucesso.")
    else:
        raise FileNotFoundError("Artefatos ausentes. Execute training/train.py primeiro.")
    
    return _modelo, _scaler

def prever(entrada: EntradaPrevisao) -> SaidaPrevisao:
    modelo, scaler = carregar_modelo()

    ultima_leitura = entrada.historico[-1].model_dump()
    features = np.array([[
        ultima_leitura["ph"],
        ultima_leitura["temperatura"],
        ultima_leitura["turbidez"],
        ultima_leitura["luminosidade"],
        ultima_leitura["radiacaoPar"]
    ]])

    features_scaled = scaler.transform(features)
    biomassa_estimada = round(float(modelo.predict(features_scaled)[0]), 2)

    LIMIAR_COLHEITA = 20.0
    TAXA_CRESCIMENTO_DIARIA = 0.5
    dias_para_colheita = 0 if biomassa_estimada >= LIMIAR_COLHEITA else int((LIMIAR_COLHEITA - biomassa_estimada) / TAXA_CRESCIMENTO_DIARIA)
    data_colheita = date.today() + timedelta(days=dias_para_colheita)

    preds_arvores = np.array([tree.predict(features_scaled)[0] for tree in modelo.estimators_])
    desvio = float(np.std(preds_arvores))
    confianca = round(max(0.0, min(100.0, 100.0 - (desvio * 15))), 1)

    status = calcular_status(biomassa_estimada)

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
                modelo="RFR_v1"
            )
    except Exception as e:
        pass # Fail silent garantido para o payload retornar mesmo sem Oracle

    return SaidaPrevisao(
        tanqueId=entrada.tanqueId,
        biomassaEstimada=biomassa_estimada,
        dataPrevistaColheita=data_colheita,
        confianca=confianca,
        status=status,
    )
