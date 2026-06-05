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
    """Carrega o modelo e o scaler para a memória RAM se ainda não estiverem ativos."""
    global _modelo, _scaler
    if _modelo is not None and _scaler is not None:
        return _modelo, _scaler

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        _modelo = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        print("Artefatos do Random Forest Regressor (RFR) carregados com sucesso.")
    else:
        raise FileNotFoundError(
            "Artefatos pkl ausentes. Execute o script training/train.py para gerar os baselines iniciais."
        )
    
    return _modelo, _scaler


def prever(entrada: EntradaPrevisao) -> SaidaPrevisao:
    modelo, scaler = carregar_modelo()

    # Extrai a última leitura contida no histórico (RFR opera sobre input tabular pontual)
    features = np.array([[
        entrada.ph,
        entrada.temperatura,
        entrada.turbidez,
        entrada.luminosidade,
        entrada.radiacaoPar
    ]])

    # Normalização bidimensional instantânea em memória
    features_scaled = scaler.transform(features)
    biomassa_estimada = round(float(modelo.predict(features_scaled)[0]), 2)

    # Regras de Negócio e Projeções
    LIMIAR_COLHEITA = 20.0
    TAXA_CRESCIMENTO_DIARIA = 0.5
    
    dias_para_colheita = (
        0 if biomassa_estimada >= LIMIAR_COLHEITA 
        else int((LIMIAR_COLHEITA - biomassa_estimada) / TAXA_CRESCIMENTO_DIARIA)
    )
    data_colheita = date.today() + timedelta(days=dias_para_colheita)

    # Cálculo estatístico real de confiança baseado na variância das árvores do ensemble
    preds_arvores = np.array([tree.predict(features_scaled)[0] for tree in modelo.estimators_])
    desvio = float(np.std(preds_arvores))
    confianca = round(max(0.0, min(100.0, 100.0 - (desvio * 15))), 1)

    status = calcular_status(biomassa_estimada)

    # Persistência Assíncrona / Fail-Silent no Oracle
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
        print(f"Sincronização Oracle ignorada em tempo de execução: {e}")

    return SaidaPrevisao(
        tanqueId=entrada.tanqueId,
        biomassaEstimada=biomassa_estimada,
        dataPrevistaColheita=data_colheita,
        confianca=confianca,
        status=status,
    )
