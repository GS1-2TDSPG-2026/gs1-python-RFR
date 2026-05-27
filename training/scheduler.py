"""
scheduler.py
------------
Re-treinamento contínuo do modelo com novos dados do Oracle.

Estratégia adotada: re-treino periódico completo com todos os dados históricos.
O modelo é substituído em memória (hot-swap) sem derrubar a API.

Como funciona:
1. A cada N horas (configurável via .env), o scheduler acorda.
2. Conta quantos registros novos existem no Oracle desde o último treino.
3. Se tiver dados suficientes (MIN_NOVOS_REGISTROS), re-treina.
4. Compara as métricas do novo modelo com o atual.
5. Se o novo for melhor (ou igual), faz o hot-swap — sem downtime.
6. Salva o novo model.pkl no disco.

Para iniciar junto com a API, importe e chame `iniciar_scheduler()` no main.py.

Dependência extra:
    pip install apscheduler==3.10.4
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from threading import Lock

from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()

# ─── Configurações (ajuste via .env) ─────────────────────────────────────────

# Intervalo entre tentativas de re-treino (em horas)
INTERVALO_HORAS: int = int(os.getenv("RETRAIN_INTERVAL_HOURS", "6"))

# Mínimo de registros novos para justificar um re-treino
MIN_NOVOS_REGISTROS: int = int(os.getenv("RETRAIN_MIN_NEW_RECORDS", "50"))

# Caminhos dos artefatos
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH   = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
META_PATH     = os.path.join(ARTIFACTS_DIR, "meta.json")  # guarda data e métricas do último treino

FEATURES = ["ph", "temperatura", "turbidez", "luminosidade", "radiacaoPar"]
TARGET   = "biomassa"

# Lock garante que dois re-treinos não rodem simultaneamente
_lock_treino = Lock()


# ─── Metadados do modelo ──────────────────────────────────────────────────────

def _ler_meta() -> dict:
    """Lê os metadados do último treino salvo em disco."""
    import json
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    # Se não existe meta, assume que nunca treinou (data bem antiga)
    return {
        "ultimo_treino": "2000-01-01T00:00:00",
        "mae":           999.0,
        "r2":            -999.0,
        "n_registros":   0,
    }


def _salvar_meta(mae: float, r2: float, n_registros: int) -> None:
    """Persiste os metadados do treino atual."""
    import json
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    meta = {
        "ultimo_treino": datetime.now().isoformat(),
        "mae":           round(mae, 4),
        "r2":            round(r2, 4),
        "n_registros":   n_registros,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📋 Metadados salvos: MAE={mae:.3f} | R²={r2:.3f} | n={n_registros}")


# ─── Núcleo do re-treinamento ─────────────────────────────────────────────────

def _treinar_com_df(df: pd.DataFrame) -> tuple[RandomForestRegressor, StandardScaler, float, float]:
    """
    Treina um novo modelo com o DataFrame fornecido.

    Retorna:
        (modelo, scaler, mae, r2)
    """
    X = df[FEATURES].values
    y = df[TARGET].values

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled  = scaler.transform(X_teste)

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

    return modelo, scaler, mae, r2


def _hot_swap_modelo(novo_modelo: RandomForestRegressor, novo_scaler: StandardScaler) -> None:
    """
    Substitui o modelo em memória usado pelo predictor.py SEM derrubar a API.

    Importa o predictor em tempo de execução para evitar importação circular,
    e substitui a variável global _modelo diretamente.
    """
    import app.models.predictor as predictor_module

    # Salva no disco
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(novo_modelo,  MODEL_PATH)
    joblib.dump(novo_scaler,  SCALER_PATH)

    # Substitui em memória (hot-swap — sem downtime)
    predictor_module._modelo = novo_modelo
    print("🔄 Hot-swap concluído — novo modelo ativo em memória")


# ─── Job principal ────────────────────────────────────────────────────────────

def job_retreinar() -> None:
    """
    Função executada periodicamente pelo scheduler.

    Fluxo:
    1. Tenta adquirir o lock (evita execuções paralelas).
    2. Conta registros novos no Oracle.
    3. Se suficiente, carrega todos os dados e re-treina.
    4. Compara métricas: só faz hot-swap se o novo modelo não for pior.
    """
    if not _lock_treino.acquire(blocking=False):
        print("⏳ Re-treino já em andamento, pulando esta rodada.")
        return

    try:
        print(f"\n{'='*55}")
        print(f"🕐 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Verificando necessidade de re-treino...")

        # Importa aqui para evitar ciclo de importação no startup
        from app.db.oracle import carregar_dados_treino, contar_registros_novos

        meta = _ler_meta()
        ultimo_treino = datetime.fromisoformat(meta["ultimo_treino"])
        novos = contar_registros_novos(ultimo_treino)

        print(f"   Registros novos desde {ultimo_treino.strftime('%Y-%m-%d %H:%M')}: {novos}")

        if novos < MIN_NOVOS_REGISTROS:
            print(f"   ℹ️  Menos que o mínimo ({MIN_NOVOS_REGISTROS}). Re-treino adiado.")
            return

        # Carrega todos os dados históricos (treino completo, não incremental)
        df = carregar_dados_treino()

        if len(df) < 30:
            print(f"   ⚠️  Apenas {len(df)} registros totais — insuficiente para treinar.")
            return

        print(f"   🔧 Treinando com {len(df)} registros...")
        novo_modelo, novo_scaler, novo_mae, novo_r2 = _treinar_com_df(df)

        # Compara com modelo atual
        mae_atual = meta.get("mae", 999.0)
        r2_atual  = meta.get("r2",  -999.0)

        print(f"   📊 Modelo atual  → MAE: {mae_atual:.3f} | R²: {r2_atual:.3f}")
        print(f"   📊 Novo modelo   → MAE: {novo_mae:.3f} | R²: {novo_r2:.3f}")

        # Aceita o novo modelo se melhorar o MAE em pelo menos 1%
        # ou se ainda não há modelo treinado com dados reais
        if novo_mae <= mae_atual * 1.01 or mae_atual == 999.0:
            _hot_swap_modelo(novo_modelo, novo_scaler)
            _salvar_meta(novo_mae, novo_r2, len(df))
            print(f"   ✅ Novo modelo aceito e ativado.")
        else:
            print(f"   ⚠️  Novo modelo é pior — mantendo o atual.")

    except Exception as e:
        print(f"   ❌ Erro durante re-treino: {e}")
    finally:
        _lock_treino.release()
        print(f"{'='*55}\n")


# ─── Inicialização ────────────────────────────────────────────────────────────

def iniciar_scheduler() -> BackgroundScheduler:
    """
    Cria e inicia o scheduler em background.
    Chame esta função no startup do FastAPI (main.py).

    Retorna o scheduler para que possa ser encerrado no shutdown se necessário.

    Exemplo de uso no main.py:
        from training.scheduler import iniciar_scheduler

        @app.on_event("startup")
        async def ao_iniciar():
            carregar_modelo()
            iniciar_scheduler()
    """
    scheduler = BackgroundScheduler(timezone="America/Sao_Paulo")

    scheduler.add_job(
        func=job_retreinar,
        trigger="interval",
        hours=INTERVALO_HORAS,
        id="retreino_modelo",
        name=f"Re-treino a cada {INTERVALO_HORAS}h",
        replace_existing=True,
        max_instances=1,           # nunca roda duas instâncias ao mesmo tempo
        next_run_time=datetime.now() + timedelta(minutes=5),  # primeira execução 5min após subir
    )

    scheduler.start()
    print(f"⏰ Scheduler ativo — re-treino a cada {INTERVALO_HORAS}h "
          f"(mínimo {MIN_NOVOS_REGISTROS} registros novos para ativar)")
    return scheduler