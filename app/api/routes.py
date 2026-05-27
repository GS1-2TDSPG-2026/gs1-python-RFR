"""
routes.py
---------
Define os endpoints (URLs) da API.
Aqui você diz: "quando alguém chamar POST /predict, execute essa função".

FastAPI cuida automaticamente de:
- Receber o JSON
- Validar com Pydantic
- Serializar a resposta de volta para JSON
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import EntradaPrevisao, SaidaPrevisao
from app.models.predictor import prever

# APIRouter é como um "grupo de rotas" — ajuda a organizar
router = APIRouter(prefix="/api/v1", tags=["Previsão de Biomassa"])


@router.post("/predict", response_model=SaidaPrevisao, summary="Prever crescimento de biomassa")
def endpoint_prever(entrada: EntradaPrevisao) -> SaidaPrevisao:
    """
    Recebe dados dos sensores IoT e dados orbitais,
    e retorna a previsão de crescimento de biomassa para as próximas 48h.

    - **tanqueId**: qual tanque está sendo analisado
    - **ph**: nível de pH da água
    - **temperatura**: temperatura em graus Celsius
    - **turbidez**: turbidez da água em NTU
    - **luminosidade**: luminosidade em lux
    - **radiacaoPar**: radiação PAR em mol/m²/dia
    """
    try:
        resultado = prever(entrada)
        return resultado
    except Exception as e:
        # Se algo der errado no modelo, retorna erro 500 com mensagem clara
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")


@router.get("/health", summary="Verificar se a API está funcionando")
def health_check() -> dict:
    """
    Endpoint simples para verificar se o serviço está no ar.
    Útil para monitoramento e para o Java/.NET saber se a IA está disponível.
    """
    return {"status": "ok", "servico": "biomassa-ia", "versao": "1.0.0"}
