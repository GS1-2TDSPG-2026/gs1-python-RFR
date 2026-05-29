import pytest
from app.models.schemas import EntradaPrevisao, SaidaPrevisao, LeituraSensor
from app.models.predictor import prever
from app.data.preprocessor import calcular_status


def _leituras(ph=7.2, temperatura=26.5, turbidez=38.0, luminosidade=820, radiacaoPar=5.7, n=7):
    return [
        LeituraSensor(ph=ph, temperatura=temperatura, turbidez=turbidez,
                      luminosidade=luminosidade, radiacaoPar=radiacaoPar)
        for _ in range(n)
    ]


def test_status_crescimento_ideal():
    assert calcular_status(16.0) == "CRESCIMENTO_IDEAL"

def test_status_crescimento_moderado():
    assert calcular_status(12.0) == "CRESCIMENTO_MODERADO"

def test_status_crescimento_lento():
    assert calcular_status(7.0) == "CRESCIMENTO_LENTO"

def test_status_alerta():
    assert calcular_status(3.0) == "ALERTA_BAIXA_BIOMASSA"


def test_previsao_retorna_saida_valida():
    entrada = EntradaPrevisao(tanqueId=1, historico=_leituras())
    resultado = prever(entrada)

    assert isinstance(resultado, SaidaPrevisao)
    assert resultado.tanqueId == 1
    assert resultado.biomassaEstimada > 0
    assert 0 <= resultado.confianca <= 100
    assert resultado.status in [
        "CRESCIMENTO_IDEAL", "CRESCIMENTO_MODERADO",
        "CRESCIMENTO_LENTO", "ALERTA_BAIXA_BIOMASSA",
    ]


def test_condicoes_ruins_geram_baixa_biomassa():
    entrada_ruim = EntradaPrevisao(
        tanqueId=2,
        historico=_leituras(ph=4.0, temperatura=18.0, turbidez=75.0, luminosidade=300, radiacaoPar=2.0)
    )
    entrada_boa = EntradaPrevisao(
        tanqueId=3,
        historico=_leituras(ph=7.5, temperatura=27.0, turbidez=20.0, luminosidade=1000, radiacaoPar=7.0)
    )

    assert prever(entrada_ruim).biomassaEstimada < prever(entrada_boa).biomassaEstimada


def test_historico_minimo_7_leituras():
    with pytest.raises(Exception):
        EntradaPrevisao(tanqueId=1, historico=_leituras(n=3))


def test_schemas_valida_ph_invalido():
    with pytest.raises(Exception):
        EntradaPrevisao(
            tanqueId=1,
            historico=[
                LeituraSensor(ph=15.0, temperatura=26.5, turbidez=38.0,
                              luminosidade=820, radiacaoPar=5.7)
                for _ in range(7)
            ]
        )
