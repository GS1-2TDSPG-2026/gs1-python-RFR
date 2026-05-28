from fastapi import FastAPI
from app.api.routes import router
from app.models.predictor import carregar_modelo

# Cria a aplicação FastAPI com metadados que aparecem na documentação
app = FastAPI(
    title="Biomassa IA — Motor Preditivo",
    description="Serviço de previsão de crescimento de biomassa usando Machine Learning.",
    version="1.0.0",
    contact={"name": "Equipe de IA"},
)

# Registra as rotas definidas em routes.py
app.include_router(router)


@app.on_event("startup")
async def ao_iniciar():
    print("Iniciando serviço de IA...")
    carregar_modelo()
    print("API pronta para receber requisições.")


@app.get("/", include_in_schema=False)
def raiz():
    return {"mensagem": "Biomassa IA está no ar. Acesse /docs para a documentação."}
