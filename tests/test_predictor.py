"""
test_predictor.py
-----------------
Testes automatizados para garantir que a IA está funcionando corretamente.
Para rodar: pytest tests/

Por que testar?
- Garante que mudanças no código não quebram o que já funciona.
- Documenta o comportamento esperado do sistema.
- Dá confiança para entregar o projeto.
"""

import pytest
from app.models.schemas import EntradaPrevisao, SaidaPrevisao
from app.models.predictor import prever
from app.data.preprocessor import calcular_status


# ─── Testes do preprocessor ───────────────────────────────────────────────────

def test_status_crescimento_ideal():
    assert calcular_status(16.0) == "CRESCIMENTO_IDEAL"

def test_status_crescimento_moderado():
    assert calcular_status(12.0) == "CRESCIMENTO_MODERADO"

def test_status_crescimento_lento():
    assert calcular_status(7.0) == "CRESCIMENTO_LENTO"

def test_status_alerta():
    assert calcular_status(3.0) == "ALERTA_BAIXA_BIOMASSA"


# ─── Testes do predictor ──────────────────────────────────────────────────────

def test_previsao_retorna_saida_valida():
    """Verifica que a previsão retorna um objeto com todos os campos"""
    entrada = EntradaPrevisao(
        tanqueId=1,
        ph=7.2,
        temperatura=26.5,
        turbidez=38.0,
        luminosidade=820,
        radiacaoPar=5.7,
    )
    resultado = prever(entrada)

    assert isinstance(resultado, SaidaPrevisao)
    assert resultado.tanqueId == 1
    assert resultado.biomassaEstimada > 0
    assert 0 <= resultado.confianca <= 100
    assert resultado.status in [
        "CRESCIMENTO_IDEAL",
        "CRESCIMENTO_MODERADO",
        "CRESCIMENTO_LENTO",
        "ALERTA_BAIXA_BIOMASSA",
    ]


def test_condicoes_ruins_geram_baixa_biomassa():
    """pH extremo e baixa luz devem resultar em biomassa mais baixa"""
    entrada_ruim = EntradaPrevisao(
        tanqueId=2,
        ph=4.0,        # pH muito ácido
        temperatura=18.0,
        turbidez=75.0,  # muito turbio
        luminosidade=300,
        radiacaoPar=2.0,
    )
    entrada_boa = EntradaPrevisao(
        tanqueId=3,
        ph=7.5,
        temperatura=27.0,
        turbidez=20.0,
        luminosidade=1000,
        radiacaoPar=7.0,
    )

    resultado_ruim = prever(entrada_ruim)
    resultado_bom = prever(entrada_boa)

    assert resultado_ruim.biomassaEstimada < resultado_bom.biomassaEstimada


def test_schemas_valida_ph_invalido():
    """pH fora do range 0-14 deve ser rejeitado pelo Pydantic"""
    with pytest.raises(Exception):
        EntradaPrevisao(
            tanqueId=1,
            ph=15.0,  # inválido
            temperatura=26.5,
            turbidez=38.0,
            luminosidade=820,
            radiacaoPar=5.7,
        )
