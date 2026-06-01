from fastapi import FastAPI
from app.api.routes import router
from app.models.predictor import carregar_modelo

app = FastAPI(
    title="Biomassa IA — Motor Preditivo",
    description="Serviço de previsão de crescimento de biomassa usando Machine Learning.",
    version="1.0.0",
    contact={"name": "Equipe de IA"},
)

app.include_router(router)


@app.on_event("startup")
async def ao_iniciar():
    print("Iniciando serviço de IA...")

    # 1. Carrega o modelo na memória
    carregar_modelo()

    # 2. Inicia o scheduler de re-treino automático
    # Intervalo configurável via .env: RETRAIN_INTERVAL_HOURS (padrão: 6)
    # Mínimo de novos registros para re-treinar: RETRAIN_MIN_NEW_RECORDS (padrão: 50)
    # Para 2h: adicione RETRAIN_INTERVAL_HOURS=2 no .env
    try:
        from training.scheduler import iniciar_scheduler
        iniciar_scheduler()
    except Exception as e:
        print(f"Scheduler não iniciado (Oracle indisponível?): {e}")

    print("API pronta para receber requisições.")


@app.get("/", include_in_schema=False)
def raiz():
    return {"mensagem": "Biomassa IA está no ar. Acesse /docs para a documentação."}

