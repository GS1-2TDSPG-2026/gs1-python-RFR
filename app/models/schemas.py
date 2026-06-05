from pydantic import BaseModel, Field
from datetime import date


class EntradaPrevisao(BaseModel):
    tanqueId: int = Field(..., description="Identificador do tanque")
    ph: float = Field(..., ge=0.0, le=14.0)
    temperatura: float = Field(..., ge=0.0, le=50.0)
    turbidez: float = Field(..., ge=0.0)
    luminosidade: float = Field(..., ge=0.0)
    radiacaoPar: float = Field(..., ge=0.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "tanqueId": 1,
                "ph": 7.2,
                "temperatura": 26.5,
                "turbidez": 38.0,
                "luminosidade": 820,
                "radiacaoPar": 5.7,
            }
        }
    }


class SaidaPrevisao(BaseModel):
    tanqueId: int
    biomassaEstimada: float = Field(..., description="Biomassa prevista em g/L")
    dataPrevistaColheita: date = Field(..., description="Data estimada de colheita")
    confianca: float = Field(..., ge=0.0, le=100.0, description="Confiança do modelo em %")
    status: str = Field(..., description="Status do crescimento")|
