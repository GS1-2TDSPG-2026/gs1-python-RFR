from fastapi import FastAPI
from app.api.routes import router
from app.models.predictor import carregar_modelo
from pathlib import Path
from training.train import treinar_modelo

app = FastAPI(
    title="Biomassa IA — Motor Preditivo",
    description="Serviço de previsão de crescimento de biomassa usando Machine Learning.",
    version="1.0.0",
)

app.include_router(router)

@app.on_event("startup")
async def ao_iniciar():
    print("Iniciando serviço de IA...")

    try:

        if (
            not Path("artifacts/model.pkl").exists()
            or
            not Path("artifacts/scaler.pkl").exists()
        ):
            print("Primeiro treinamento iniciado...")
            treinar_modelo()

        carregar_modelo()

    except Exception as e:
        print(f"Erro ao iniciar modelo: {e}")
        raise

    try:
        from training.scheduler import iniciar_scheduler
        iniciar_scheduler()
    except Exception as e:
        print(f"Scheduler não iniciado: {e}")

    print("API pronta para receber requisições.")

@app.get("/", include_in_schema=False)
def raiz():
    return {"mensagem": "Biomassa IA operando via RFR. Acesse /docs."}
