import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from threading import Lock

from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv

from app.data.preprocessor import criar_janelas, FEATURES, JANELA

load_dotenv()

INTERVALO_HORAS: int = int(os.getenv("RETRAIN_INTERVAL_HOURS", "6"))
MIN_NOVOS_REGISTROS: int = int(os.getenv("RETRAIN_MIN_NEW_RECORDS", "50"))

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "model_lstm.keras")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
META_PATH   = os.path.join(ARTIFACTS_DIR, "meta.json")

_lock_treino = Lock()


def _ler_meta() -> dict:
    import json
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {"ultimo_treino": "2000-01-01T00:00:00", "mae": 999.0, "r2": -999.0, "n_registros": 0}


def _salvar_meta(mae: float, r2: float, n_registros: int) -> None:
    import json
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump({
            "ultimo_treino": datetime.now().isoformat(),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "n_registros": n_registros,
        }, f, indent=2)


def _treinar_com_df(df: pd.DataFrame):
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    X, y = criar_janelas(df, JANELA)

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = Sequential([
        LSTM(64, input_shape=(JANELA, len(FEATURES)), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    modelo.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    modelo.fit(X_treino, y_treino, epochs=100, batch_size=16,
               validation_split=0.2, callbacks=[early_stop], verbose=0)

    y_pred = modelo.predict(X_teste, verbose=0).flatten()
    mae = mean_absolute_error(y_teste, y_pred)
    r2  = r2_score(y_teste, y_pred)

    return modelo, scaler, mae, r2


def _hot_swap_modelo(novo_modelo, novo_scaler) -> None:
    import app.models.predictor as predictor_module

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    novo_modelo.save(MODEL_PATH)
    joblib.dump(novo_scaler, SCALER_PATH)

    predictor_module._modelo = novo_modelo
    print("Hot-swap concluído — novo modelo LSTM ativo em memória")


def job_retreinar() -> None:
    if not _lock_treino.acquire(blocking=False):
        print("Re-treino já em andamento, pulando.")
        return

    try:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Verificando re-treino...")

        from app.db.oracle import carregar_dados_treino, contar_registros_novos

        meta = _ler_meta()
        ultimo_treino = datetime.fromisoformat(meta["ultimo_treino"])
        novos = contar_registros_novos(ultimo_treino)

        print(f"Registros novos: {novos}")

        if novos < MIN_NOVOS_REGISTROS:
            print(f"Abaixo do mínimo ({MIN_NOVOS_REGISTROS}). Re-treino adiado.")
            return

        df = carregar_dados_treino()

        if len(df) < JANELA + 10:
            print(f"Dados insuficientes ({len(df)} registros).")
            return

        print(f"Treinando com {len(df)} registros...")
        novo_modelo, novo_scaler, novo_mae, novo_r2 = _treinar_com_df(df)

        mae_atual = meta.get("mae", 999.0)

        print(f"Modelo atual → MAE: {mae_atual:.3f}")
        print(f"Novo modelo  → MAE: {novo_mae:.3f} | R²: {novo_r2:.3f}")

        if novo_mae <= mae_atual * 1.01 or mae_atual == 999.0:
            _hot_swap_modelo(novo_modelo, novo_scaler)
            _salvar_meta(novo_mae, novo_r2, len(df))
            print("Novo modelo aceito.")
        else:
            print("Novo modelo é pior — mantendo o atual.")

    except Exception as e:
        print(f"Erro durante re-treino: {e}")
    finally:
        _lock_treino.release()


def iniciar_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="America/Sao_Paulo")
    scheduler.add_job(
        func=job_retreinar,
        trigger="interval",
        hours=INTERVALO_HORAS,
        id="retreino_lstm",
        replace_existing=True,
        max_instances=1,
        next_run_time=datetime.now() + timedelta(minutes=5),
    )
    scheduler.start()
    print(f"Scheduler ativo — re-treino a cada {INTERVALO_HORAS}h")
    return scheduler
