"""
train.py
--------
Script de TREINAMENTO do modelo. Você roda isso UMA VEZ (ou periodicamente)
para gerar o arquivo model.pkl que a API usa.

Fluxo:
1. Lê histórico do Oracle (ou CSV para testes)
2. Separa features (entradas) do target (saída = biomassa)
3. Normaliza os dados
4. Treina o Random Forest
5. Avalia a qualidade do modelo (MAE, R²)
6. Salva model.pkl e scaler.pkl na pasta artifacts/

Para rodar: python -m training.train
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ─── Configurações ────────────────────────────────────────────────────────────

FEATURES = ["ph", "temperatura", "turbidez", "luminosidade", "radiacaoPar"]
TARGET = "biomassa"

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

# ─── Funções ──────────────────────────────────────────────────────────────────

def carregar_dados_csv(caminho: str) -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV para treino.
    Use isso no início, antes de ter o Oracle configurado.

    O CSV deve ter as colunas: ph, temperatura, turbidez, luminosidade, radiacaoPar, biomassa
    """
    df = pd.read_csv(caminho)
    print(f"📂 CSV carregado: {len(df)} registros")
    return df


def carregar_dados_oracle() -> pd.DataFrame:
    """
    Carrega histórico de crescimento direto do Oracle.
    Descomente e configure quando o banco estiver disponível.
    """
    import oracledb
    from dotenv import load_dotenv
    load_dotenv()

    conn = oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN"),
    )

    query = """
        SELECT
            m.ph,
            m.temperatura,
            m.turbidez,
            m.luminosidade,
            m.radiacao_par   AS radiacaoPar,
            c.biomassa_final AS biomassa
        FROM
            medicoes_sensor m
            JOIN colheitas c ON m.tanque_id = c.tanque_id
                             AND m.data_medicao = c.data_referencia
        WHERE
            c.biomassa_final IS NOT NULL
        ORDER BY
            m.data_medicao
    """

    df = pd.read_sql(query, con=conn)
    conn.close()
    print(f"🗄️  Oracle: {len(df)} registros carregados")
    return df


def treinar(df: pd.DataFrame) -> None:
    """
    Treina o modelo Random Forest e salva os artefatos.
    """
    # 1. Separa X (features) de y (o que queremos prever)
    X = df[FEATURES].values
    y = df[TARGET].values

    # 2. Divide em treino (80%) e teste (20%)
    #    O conjunto de teste serve para avaliar o modelo com dados que ele nunca viu
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Normaliza — coloca tudo na mesma escala
    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)  # aprende a escala do treino
    X_teste_scaled = scaler.transform(X_teste)         # aplica a mesma escala no teste

    # 4. Treina o modelo
    print("🔧 Treinando Random Forest...")
    modelo = RandomForestRegressor(
        n_estimators=100,   # 100 árvores na floresta
        max_depth=10,       # profundidade máxima de cada árvore
        random_state=42,    # garante resultados reproduzíveis
        n_jobs=-1,          # usa todos os núcleos do processador
    )
    modelo.fit(X_treino_scaled, y_treino)

    # 5. Avalia o modelo
    y_pred = modelo.predict(X_teste_scaled)

    mae = mean_absolute_error(y_teste, y_pred)
    r2 = r2_score(y_teste, y_pred)

    print(f"\n📊 Métricas do modelo:")
    print(f"   MAE (Erro Médio Absoluto): {mae:.3f} g/L")
    print(f"   R² (Coeficiente de Determinação): {r2:.3f}")
    print(f"   → R² próximo de 1.0 = modelo bom | próximo de 0 = modelo ruim")

    # 6. Importância das features — quais variáveis mais influenciam a previsão
    importancias = pd.Series(modelo.feature_importances_, index=FEATURES)
    print(f"\n🌟 Importância das variáveis:")
    for feature, imp in importancias.sort_values(ascending=False).items():
        print(f"   {feature}: {imp:.3f}")

    # 7. Salva os artefatos
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n💾 Modelo salvo em: {MODEL_PATH}")
    print(f"💾 Scaler salvo em: {SCALER_PATH}")


# ─── Execução ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Para testar, gera dados sintéticos se não tiver CSV ou Oracle ainda
    print("🌱 Gerando dados sintéticos para treino de demonstração...")

    np.random.seed(42)
    n = 500  # 500 amostras sintéticas

    df_demo = pd.DataFrame({
        "ph":           np.random.uniform(6.0, 9.0, n),
        "temperatura":  np.random.uniform(18.0, 35.0, n),
        "turbidez":     np.random.uniform(10.0, 80.0, n),
        "luminosidade": np.random.uniform(400.0, 1200.0, n),
        "radiacaoPar":  np.random.uniform(3.0, 9.0, n),
    })

    # Biomassa simulada com relação realista às variáveis
    # (em produção isso vem do histórico real do Oracle)
    df_demo["biomassa"] = (
        2.0 * df_demo["radiacaoPar"]
        + 0.3 * df_demo["luminosidade"] / 100
        - 0.5 * abs(df_demo["ph"] - 7.5)
        - 0.1 * df_demo["turbidez"]
        + 0.2 * df_demo["temperatura"]
        + np.random.normal(0, 0.5, n)  # ruído
    ).clip(1.0, 25.0)

    treinar(df_demo)
    print("\n✅ Treinamento concluído!")
