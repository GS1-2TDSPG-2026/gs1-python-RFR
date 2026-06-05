import os
import joblib
import numpy as np
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.train import treinar_modelo, FEATURES, TARGET, MODEL_PATH, SCALER_PATH
from app.db.oracle import contar_registros_novos, carregar_dados_treino

scheduler_global = None
ULTIMO_TREINO = datetime.now()


def _hot_swap(novo_modelo, novo_scaler):
    import app.models.predictor as predictor
    predictor._modelo = novo_modelo
    predictor._scaler = novo_scaler
    print("[Scheduler] Modelo atualizado em memória")


def treinar_novamente():
    global ULTIMO_TREINO
    print("[Scheduler] Iniciando retreinamento...")
    try:
        df = carregar_dados_treino()
        X = df[FEATURES].values
        y = df[TARGET].values

        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        modelo = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        modelo.fit(X_train, y_train)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(modelo, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        _hot_swap(modelo, scaler)
        ULTIMO_TREINO = datetime.now()
        print("[Scheduler] Retreinamento concluído")
    except Exception as e:
        print(f"[Scheduler] Erro no retreinamento: {e}")


def verificar_dados():
    global ULTIMO_TREINO
    try:
        novos = contar_registros_novos(ULTIMO_TREINO)
        print(f"[Scheduler] Novos registros: {novos}")
        if novos >= 20:
            treinar_novamente()
    except Exception as e:
        print(f"[Scheduler] Erro: {e}")


def iniciar_scheduler():
    global scheduler_global
    scheduler_global = BackgroundScheduler()
    scheduler_global.add_job(
        verificar_dados,
        trigger="interval",
        hours=2,
        id="retreinamento_rfr",
        replace_existing=True,
    )
    scheduler_global.start()
    print("[Scheduler] Iniciado (intervalo: 2h)")