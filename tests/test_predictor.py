import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from app.models.schemas import EntradaPrevisao
from app.models.predictor import prever

@pytest.fixture
def payload_valido() -> EntradaPrevisao:
    # Simula a estrutura do schema instanciado com o histórico do sensor
    mock_leitura = MagicMock()
    mock_leitura.model_dump.return_value = {
        "ph": 7.2,
        "temperatura": 26.5,
        "turbidez": 38.0,
        "luminosidade": 820.0,
        "radiacaoPar": 5.7
    }
    
    mock_entrada = MagicMock(spec=EntradaPrevisao)
    mock_entrada.tanqueId = 1
    mock_entrada.historico = [mock_leitura]
    
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
    
    mock_buscar.assert_called_once_with(1)
    mock_salvar.assert_called_once()

@patch("app.db.oracle.buscar_ultimo_dado_orbital", side_effect=Exception("Timeout no Oracle"))
def test_prever_garante_retorno_mesmo_com_falha_de_banco(mock_buscar, payload_valido):
    # O pipeline não deve quebrar caso o Oracle esteja indisponível
    resultado = prever(payload_valido)
    
    assert resultado.tanqueId == 1
    assert resultado.biomassaEstimada > 0.0
