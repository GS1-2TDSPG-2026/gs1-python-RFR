from fastapi import APIRouter, HTTPException
from app.models.schemas import EntradaPrevisao, SaidaPrevisao
from app.models.predictor import prever

# APIRouter é como um "grupo de rotas" — ajuda a organizar
router = APIRouter(prefix="/api/v1", tags=["Previsão de Biomassa"])


@router.post("/predict", response_model=SaidaPrevisao, summary="Prever crescimento de biomassa")
def endpoint_prever(entrada: EntradaPrevisao) -> SaidaPrevisao:
    try:
        resultado = prever(entrada)
        return resultado
    except Exception as e:
        # Se algo der errado no modelo, retorna erro 500 com mensagem clara
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")


@router.get("/health", summary="Verificar se a API está funcionando")
def health_check() -> dict:
    return {
    "tanqueId": entrada.tanqueId,

    "dadosEntrada": {
        "ph": entrada.ph,
        "temperatura": entrada.temperatura,
        "turbidez": entrada.turbidez,
        "luminosidade": entrada.luminosidade,
        "radiacaoPar": entrada.radiacaoPar,
    },

    "resultado": {
        "biomassaEstimada": biomassa,
        "dataPrevistaColheita": data_colheita,
        "confianca": confianca,
        "status": status,
    },
}
