import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from app.models.schemas import EntradaPrevisao
from app.models.predictor import prever

@pytest.fixture
def payload_valido() -> EntradaPrevisao:
    mock_entrada = MagicMock(spec=EntradaPrevisao)
    mock_entrada.tanqueId = 1
    mock_entrada.ph = 7.2
    mock_entrada.temperatura = 26.5
    mock_entrada.turbidez = 38.0
    mock_entrada.luminosidade = 820.0
    mock_entrada.radiacaoPar = 5.7
    return mock_entrada

@patch("app.db.oracle.buscar_ultimo_dado_orbital")
@patch("app.db.oracle.salvar_previsao")
def test_prever_retorna_saida_valida_e_grava_no_oracle(mock_salvar, mock_buscar, payload_valido):
    mock_buscar.return_value = 100
    
    resultado = prever(payload_valido)
    
    assert resultado.tanqueId == 1
    assert isinstance(resultado.biomassaEstimada, float)
    assert isinstance(resultado.confianca, float)
    assert resultado.status in [
        "CRESCIMENTO_IDEAL", "CRESCIMENTO_MODERADO", "CRESCIMENTO_LENTO", "ALERTA_BAIXA_BIOMASSA"
    ]
    assert resultado.dataPrevistaColheita >= date.today()
    mock_salvar.assert_called_once()

@patch("app.db.oracle.buscar_ultimo_dado_orbital", side_effect=Exception("Timeout"))
def test_prever_garante_retorno_mesmo_com_falha_de_banco(mock_buscar, payload_valido):
    resultado = prever(payload_valido)
    
    assert resultado.tanqueId == 1
    assert resultado.biomassaEstimada > 0.0
