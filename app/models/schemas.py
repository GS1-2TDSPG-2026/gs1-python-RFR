from pydantic import BaseModel, Field
from datetime import date
from typing import List


class LeituraSensor(BaseModel):
    ph: float = Field(..., ge=0.0, le=14.0)
    temperatura: float = Field(..., ge=0.0, le=50.0)
    turbidez: float = Field(..., ge=0.0)
    luminosidade: float = Field(..., ge=0.0)
    radiacaoPar: float = Field(..., ge=0.0)


class EntradaPrevisao(BaseModel):
    tanqueId: int = Field(..., description="Identificador do tanque")
    historico: List[LeituraSensor] = Field(
        ...,
        min_length=7,
        max_length=30,
        description="Sequência de leituras dos sensores (mínimo 7 dias)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "tanqueId": 1,
                "historico": [
                    {"ph": 7.0, "temperatura": 25.0, "turbidez": 32.0, "luminosidade": 790, "radiacaoPar": 5.2},
                    {"ph": 7.1, "temperatura": 25.5, "turbidez": 34.0, "luminosidade": 800, "radiacaoPar": 5.4},
                    {"ph": 7.1, "temperatura": 26.0, "turbidez": 35.0, "luminosidade": 805, "radiacaoPar": 5.5},
                    {"ph": 7.2, "temperatura": 26.2, "turbidez": 36.0, "luminosidade": 810, "radiacaoPar": 5.5},
                    {"ph": 7.2, "temperatura": 26.3, "turbidez": 37.0, "luminosidade": 815, "radiacaoPar": 5.6},
                    {"ph": 7.2, "temperatura": 26.4, "turbidez": 37.5, "luminosidade": 818, "radiacaoPar": 5.6},
                    {"ph": 7.2, "temperatura": 26.5, "turbidez": 38.0, "luminosidade": 820, "radiacaoPar": 5.7},
                ]
            }
        }
    }


class SaidaPrevisao(BaseModel):
    tanqueId: int
    biomassaEstimada: float = Field(..., description="Biomassa prevista em g/L")
    dataPrevistaColheita: date = Field(..., description="Data estimada de colheita")
    confianca: float = Field(..., ge=0.0, le=100.0, description="Confiança do modelo em %")
    status: str = Field(..., description="Status do crescimento")
