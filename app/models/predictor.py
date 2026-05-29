import os
import numpy as np
from datetime import date, timedelta

from app.data.preprocessor import preparar_sequencia, calcular_status
from app.models.schemas import EntradaPrevisao, SaidaPrevisao

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../artifacts/model_lstm.keras")

_modelo = None


def carregar_modelo():
    global _modelo

    if _modelo is not None:
        return _modelo

    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        _modelo = load_model(MODEL_PATH)
        print(f"Modelo LSTM carregado: {MODEL_PATH}")
    else:
        print("Modelo não encontrado. Usando modelo de demonstração.")
        _modelo = _criar_modelo_demo()

    return _modelo


def _criar_modelo_demo():
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from app.data.preprocessor import criar_janelas, JANELA, FEATURES

    np.random.seed(42)
    n = 300

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
        - 0.5 * abs(df["ph"] - 7.5)
        - 0.1 * df["turbidez"]
        + 0.2 * df["temperatura"]
        + np.random.normal(0, 0.5, n)
    ).clip(1.0, 25.0)

    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    artifacts_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(scaler, MODEL_PATH.replace("model_lstm.keras", "scaler.pkl"))

    X, y = criar_janelas(df, JANELA)

    modelo = Sequential([
        LSTM(64, input_shape=(JANELA, len(FEATURES)), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    modelo.compile(optimizer="adam", loss="mse")
    modelo.fit(X, y, epochs=20, batch_size=16, validation_split=0.2, verbose=0)
    modelo.save(MODEL_PATH)

    return modelo


def prever(entrada: EntradaPrevisao) -> SaidaPrevisao:
    modelo = carregar_modelo()

    historico = [d.model_dump() for d in entrada.historico]
    X = preparar_sequencia(historico)

    biomassa_estimada: float = round(float(modelo.predict(X, verbose=0)[0][0]), 2)

    LIMIAR_COLHEITA = 20.0
    TAXA_CRESCIMENTO_DIARIA = 0.5

    dias_para_colheita = (
        0 if biomassa_estimada >= LIMIAR_COLHEITA
        else int((LIMIAR_COLHEITA - biomassa_estimada) / TAXA_CRESCIMENTO_DIARIA)
    )
    data_colheita = date.today() + timedelta(days=dias_para_colheita)

    N_PRED = 10
    preds = np.array([float(modelo.predict(X, verbose=0)[0][0]) for _ in range(N_PRED)])
    desvio = float(np.std(preds))
    confianca = round(max(0.0, min(100.0, 100.0 - (desvio * 20))), 1)

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
                modelo="LSTM_v1",
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
