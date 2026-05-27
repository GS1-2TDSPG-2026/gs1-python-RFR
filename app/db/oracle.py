"""
oracle.py
---------
Gerencia a conexão com o banco Oracle e salva os resultados de previsão.

Por que salvar no banco?
- O histórico de previsões pode ser usado para retreinar o modelo futuramente.
- O dashboard do produtor pode ler esses dados.
- Permite rastrear a evolução da biomassa por tanque ao longo do tempo.
"""

import os
import oracledb
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def get_conexao():
    """
    Retorna uma conexão com o Oracle.
    As credenciais vêm do arquivo .env (nunca coloque senhas no código!).
    """
    return oracledb.connect(
        user=os.getenv("ORACLE_USER", "system"),
        password=os.getenv("ORACLE_PASSWORD", "oracle"),
        dsn=os.getenv("ORACLE_DSN", "localhost:1521/XEPDB1"),
    )


def salvar_previsao(tanque_id: int, biomassa: float, data_colheita, confianca: float, status: str) -> None:
    """
    Salva o resultado de uma previsão na tabela PREVISOES_BIOMASSA do Oracle.

    A tabela deve existir no banco (veja o SQL de criação abaixo).
    """
    sql = """
        INSERT INTO previsoes_biomassa
            (tanque_id, biomassa_estimada, data_prevista_colheita, confianca, status, criado_em)
        VALUES
            (:tanque_id, :biomassa, :data_colheita, :confianca, :status, :criado_em)
    """

    try:
        with get_conexao() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, {
                    "tanque_id":     tanque_id,
                    "biomassa":      biomassa,
                    "data_colheita": data_colheita,
                    "confianca":     confianca,
                    "status":        status,
                    "criado_em":     datetime.now(),
                })
            conn.commit()
    except Exception as e:
        # Log do erro mas não interrompe a API — a previsão já foi feita
        print(f"⚠️  Erro ao salvar no Oracle: {e}")


"""
SQL para criar a tabela no Oracle (rode uma vez no banco):

CREATE TABLE previsoes_biomassa (
    id                     NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tanque_id              NUMBER NOT NULL,
    biomassa_estimada      NUMBER(10, 2),
    data_prevista_colheita DATE,
    confianca              NUMBER(5, 2),
    status                 VARCHAR2(50),
    criado_em              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
