import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from app.data.preprocessor import criar_janelas, FEATURES, JANELA

TARGET = "biomassa"

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "model_lstm.keras")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")


def carregar_dados_csv(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    print(f"CSV carregado: {len(df)} registros")
    return df


def carregar_dados_oracle() -> pd.DataFrame:
    from app.db.oracle import get_conexao

    query = """
        SELECT
            m.ph,
            m.temperatura,
            m.turbidez,
            m.luminosidade,
            d.irradiancia_par   AS "radiacaoPar",
            p.biomassa_g_l      AS "biomassa"
        FROM
            TB_METRICAS_TANQUE  m
            JOIN TB_TANQUE       t  ON t.id_tanque    = m.id_tanque
            JOIN TB_DADO_ORBITAL d  ON d.id_fazenda   = t.id_fazenda
                                    AND d.dt_coleta BETWEEN
                                        TRUNC(CAST(m.dt_leitura AS DATE)) - 1
                                        AND TRUNC(CAST(m.dt_leitura AS DATE)) + 1
            JOIN TB_PREVISOES_IA p  ON p.id_tanque        = m.id_tanque
                                    AND p.id_dado_orbital  = d.id_dado_orbital
        WHERE
            m.ph               IS NOT NULL
            AND m.temperatura   IS NOT NULL
            AND m.turbidez      IS NOT NULL
            AND m.luminosidade  IS NOT NULL
            AND d.irradiancia_par IS NOT NULL
            AND p.biomassa_g_l  IS NOT NULL
            AND p.biomassa_g_l  > 0
        ORDER BY
            m.dt_leitura
    """
    with get_conexao() as conn:
        df = pd.read_sql(query, con=conn)
    print(f"Oracle: {len(df)} registros carregados")
    return df


def treinar(df: pd.DataFrame) -> None:
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    X, y = criar_janelas(df, JANELA)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = Sequential([
        LSTM(64, input_shape=(JANELA, len(FEATURES)), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    modelo.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    print("Treinando LSTM...")
    modelo.fit(
        X_treino, y_treino,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred = modelo.predict(X_teste, verbose=0).flatten()
    mae = mean_absolute_error(y_teste, y_pred)
    r2  = r2_score(y_teste, y_pred)

    print(f"\nMAE: {mae:.3f} g/L | R²: {r2:.3f}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    modelo.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n{MODEL_PATH}\n{SCALER_PATH}")


if __name__ == "__main__":
    np.random.seed(42)
    n = 500

    df_demo = pd.DataFrame({
        "ph":           np.random.uniform(6.0, 9.0, n),
        "temperatura":  np.random.uniform(18.0, 35.0, n),
        "turbidez":     np.random.uniform(10.0, 80.0, n),
        "luminosidade": np.random.uniform(400.0, 1200.0, n),
        "radiacaoPar":  np.random.uniform(3.0, 9.0, n),
    })
    df_demo["biomassa"] = (
        2.0 * df_demo["radiacaoPar"]
        + 0.3 * df_demo["luminosidade"] / 100
        - 0.5 * abs(df_demo["ph"] - 7.5)
        - 0.1 * df_demo["turbidez"]
        + 0.2 * df_demo["temperatura"]
        + np.random.normal(0, 0.5, n)
    ).clip(1.0, 25.0)

    treinar(df_demo)
    print("\nTreinamento concluído!")
