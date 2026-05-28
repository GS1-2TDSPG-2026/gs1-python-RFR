import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = ["ph", "temperatura", "turbidez", "luminosidade", "radiacaoPar"]
TARGET = "biomassa"

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "model.pkl")
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
    X = df[FEATURES].values
    y = df[TARGET].values

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled  = scaler.transform(X_teste)

    print("🔧 Treinando Random Forest...")
    modelo = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_treino_scaled, y_treino)

    y_pred = modelo.predict(X_teste_scaled)
    mae = mean_absolute_error(y_teste, y_pred)
    r2  = r2_score(y_teste, y_pred)

    print(f"\nMAE: {mae:.3f} g/L | R²: {r2:.3f}")

    importancias = pd.Series(modelo.feature_importances_, index=FEATURES)
    print("\nImportância das variáveis:")
    for feat, imp in importancias.sort_values(ascending=False).items():
        print(f"   {feat}: {imp:.3f}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
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
