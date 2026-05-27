import os
from datetime import datetime
import oracledb
from dotenv import load_dotenv

load_dotenv()


def get_conexao():
    return oracledb.connect(
        user=os.getenv("ORACLE_USER", "system"),
        password=os.getenv("ORACLE_PASSWORD", "oracle"),
        dsn=os.getenv("ORACLE_DSN", "localhost:1521/XEPDB1"),
    )


def salvar_previsao(
    tanque_id: int,
    id_dado_orbital: int | None,
    biomassa: float,
    data_pico,
    confianca: float,
) -> None:
    sql = """
        INSERT INTO TB_PREVISOES_IA
            (id_tanque, id_dado_orbital, dt_previsao, biomassa_g_l, dt_pico_previsto, confianca_pct, modelo_utilizado)
        VALUES
            (:tanque_id, :id_dado_orbital, :dt_previsao, :biomassa, :data_pico, :confianca, 'RandomForest_v1')
    """
    try:
        with get_conexao() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, {
                    "tanque_id":       tanque_id,
                    "id_dado_orbital": id_dado_orbital,
                    "dt_previsao":     datetime.now(),
                    "biomassa":        biomassa,
                    "data_pico":       data_pico,
                    "confianca":       confianca,
                })
            conn.commit()
    except Exception as e:
        print(f"⚠️  Erro ao salvar previsão: {e}")


def salvar_alerta(
    tanque_id: int,
    id_metrica: int | None,
    tipo_alerta: str,
    mensagem: str,
) -> None:
    sql = """
        INSERT INTO TB_ALERTA_CRITICO
            (id_tanque, id_metrica, tipo_alerta, severidade, mensagem, status, dt_alerta)
        VALUES
            (:tanque_id, :id_metrica, :tipo_alerta, 'ALTA', :mensagem, 'ABERTO', CURRENT_TIMESTAMP)
    """
    try:
        with get_conexao() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, {
                    "tanque_id":   tanque_id,
                    "id_metrica":  id_metrica,
                    "tipo_alerta": tipo_alerta,
                    "mensagem":    mensagem,
                })
            conn.commit()
    except Exception as e:
        print(f"⚠️  Erro ao salvar alerta: {e}")