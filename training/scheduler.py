import os
import joblib
from datetime import datetime, timedelta
from threading import Lock

from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()

INTERVALO_HORAS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "2"))
MIN_NOVOS_REGISTROS = int(os.getenv("RETRAIN_MIN_NEW_RECORDS", "50"))
FEATURES = ["ph", "temperatura", "turbidez", "luminosidade", "radiacaoPar"]

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")

_lock_treino = Lock()

def _ler_meta() -> dict:
    import json
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {"ultimo_treino": "2000-01-01T00:00:00", "mae": 999.0, "r2": -999.0}

def _salvar_meta(mae: float, r2: float, n: int) -> None:
    import json
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump({
            "ultimo_treino": datetime.now().isoformat(),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "n_registros": n,
        }, f, indent=2)

def _hot_swap(novo_modelo, novo_scaler) -> None:
    import app.models.predictor as predictor_module
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(novo_modelo, MODEL_PATH)
    joblib.dump(novo_scaler, SCALER_PATH)

    predictor_module._modelo = novo_modelo
    predictor_module._scaler = novo_scaler
    print("Hot-swap concluído — novo Random Forest e Scaler ativos em memória")

def job_retreinar() -> None:
    if not _lock_treino.acquire(blocking=False): return
    
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Verificando re-treino...")
        from app.db.oracle import carregar_dados_treino, contar_registros_novos
        
        meta = _ler_meta()
        ultimo = datetime.fromisoformat(meta["ultimo_treino"])
        novos = contar_registros_novos(ultimo)

        if novos < MIN_NOVOS_REGISTROS:
            print(f"Adiado. {novos} registros novos (Mínimo: {MIN_NOVOS_REGISTROS})")
            return

        df = carregar_dados_treino()
        if len(df) < 20: return

        X = df[FEATURES].values
        y = df["biomassa"].values
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_treino_s = scaler.fit_transform(X_treino)
        X_teste_s = scaler.transform(X_teste)

        modelo = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        modelo.fit(X_treino_s, y_treino)

        novo_mae = mean_absolute_error(y_teste, modelo.predict(X_teste_s))
        novo_r2 = r2_score(y_teste, modelo.predict(X_teste_s))
        mae_atual = meta.get("mae", 999.0)

        if novo_mae <= mae_atual * 1.01 or mae_atual == 999.0:
            _hot_swap(modelo, scaler)
            _salvar_meta(novo_mae, novo_r2, len(df))
            print(f"Novo modelo aceito. MAE: {novo_mae:.3f}")
        else:
            print("Novo modelo descartado por performance inferior.")

    except Exception as e:
        print(f"Erro no pipeline de treino: {e}")
    finally:
        _lock_treino.release()

def iniciar_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="America/Sao_Paulo")
    scheduler.add_job(
        func=job_retreinar,
        trigger="interval",
        hours=INTERVALO_HORAS, # Mude para minutes=2 no seu teste local
        id="retreino_rf",
        replace_existing=True,
        max_instances=1,
        next_run_time=datetime.now() + timedelta(minutes=1)
    )
    scheduler.start()
    print(f"Scheduler ativo — ciclo configurado para {INTERVALO_HORAS}h")
    return scheduler
