"""
schemas.py
----------
Define o "contrato" da API: o que ela recebe e o que ela devolve.
Pydantic valida automaticamente os tipos e lança erros claros se algo vier errado.
"""

from pydantic import BaseModel, Field
from datetime import date


class EntradaPrevisao(BaseModel):
    """Dados que chegam na requisição POST /predict"""

    tanqueId: int = Field(..., description="Identificador do tanque")
    ph: float = Field(..., ge=0.0, le=14.0, description="Nível de pH (0 a 14)")
    temperatura: float = Field(..., ge=0.0, le=50.0, description="Temperatura em °C")
    turbidez: float = Field(..., ge=0.0, description="Turbidez em NTU")
    luminosidade: float = Field(..., ge=0.0, description="Luminosidade em lux")
    radiacaoPar: float = Field(..., ge=0.0, description="Radiação PAR em mol/m²/dia")

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
    """Dados devolvidos pela API após a previsão"""

    tanqueId: int
    biomassaEstimada: float = Field(..., description="Biomassa prevista em g/L")
    dataPrevistaColheita: date = Field(..., description="Data estimada de colheita")
    confianca: float = Field(..., ge=0.0, le=100.0, description="Confiança do modelo em %")
    status: str = Field(..., description="Status do crescimento")
