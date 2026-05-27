# 🌱 Biomassa IA — Motor Preditivo de Crescimento de Microalgas

API de Machine Learning para previsão de crescimento de biomassa em tanques de cultivo, com re-treinamento contínuo e integração ao Oracle Database.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Executando a API](#executando-a-api)
- [Endpoints](#endpoints)
- [Treinamento do Modelo](#treinamento-do-modelo)
- [Re-treinamento Contínuo](#re-treinamento-contínuo)
- [Banco de Dados Oracle](#banco-de-dados-oracle)
- [Testes](#testes)
- [Deploy na OCI](#deploy-na-oci)
- [Estrutura do Projeto](#estrutura-do-projeto)

---

## Visão Geral

Este serviço expõe uma API REST que recebe dados de sensores IoT (pH, temperatura, turbidez, luminosidade, radiação PAR) e retorna a previsão de biomassa (g/L) para as próximas 48h, a data estimada de colheita e a confiança do modelo.

O modelo é um **Random Forest Regressor** (scikit-learn) re-treinado automaticamente conforme novos dados reais de colheita chegam ao Oracle.

**Stack:**
- Python 3.11+
- FastAPI + Uvicorn
- scikit-learn (Random Forest)
- Oracle Database (via `python-oracledb`)
- APScheduler (re-treinamento agendado)

---

## Arquitetura

```
Sensores IoT / App Java
        │
        ▼ POST /api/v1/predict
┌───────────────────┐
│   FastAPI (8000)  │
│  ┌─────────────┐  │
│  │ predictor   │◄─┼── model.pkl + scaler.pkl (carregados na memória)
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │  scheduler  │  │──► a cada N horas: lê Oracle → re-treina → hot-swap
│  └─────────────┘  │
└───────────────────┘
        │
        ▼ INSERT
┌───────────────────┐
│  Oracle Database  │
│  medicoes_sensor  │
│  colheitas        │
│  previsoes_biom.  │
└───────────────────┘
```

O hot-swap substitui o modelo em memória **sem derrubar a API** — não há downtime durante o re-treinamento.

---

## Pré-requisitos

- Python 3.11 ou superior
- Oracle Database acessível (local ou na nuvem)
- Oracle Instant Client **não é necessário** com `python-oracledb` no modo Thin (padrão)

Verifique sua versão do Python:
```bash
python --version
```

---

## Instalação

```bash
# Clone o repositório
git clone https://github.com/GS1-2TDSPG-2026/gs1-python-LSTM.git
cd biomassa-ia

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Instale as dependências
pip install -r requirements.txt

# Instale o APScheduler (re-treinamento contínuo)
pip install apscheduler==3.10.4
```

---

## Configuração

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o `.env`:

```env
# ── Oracle ───────────────────────────────────────────
ORACLE_USER=system
ORACLE_PASSWORD=sua_senha_aqui
ORACLE_DSN=localhost:1521/XEPDB1
# Para Oracle Cloud (ADB): ORACLE_DSN=adb.sa-saopaulo-1.oraclecloud.com:1522/sua_string

# ── Re-treinamento ───────────────────────────────────
RETRAIN_INTERVAL_HOURS=6        # a cada 6 horas o scheduler verifica
RETRAIN_MIN_NEW_RECORDS=50      # mínimo de novos registros para re-treinar
```

> **Nunca** commite o arquivo `.env`. Ele já está no `.gitignore`.

---

## Executando a API

### Desenvolvimento (com hot-reload)

```bash
uvicorn app.main:app --reload --port 8000
```

### Produção

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> Use `--workers 1` enquanto o modelo está em memória compartilhada. Para múltiplos workers, o modelo precisaria ser carregado de um servidor de modelos (ex: MLflow).

Acesse a documentação interativa em: **http://localhost:8000/docs**

---

## Endpoints

### `POST /api/v1/predict`

Recebe os dados dos sensores e retorna a previsão.

**Request body:**
```json
{
  "tanqueId": 1,
  "ph": 7.2,
  "temperatura": 26.5,
  "turbidez": 38.0,
  "luminosidade": 820,
  "radiacaoPar": 5.7
}
```

**Response `200 OK`:**
```json
{
  "tanqueId": 1,
  "biomassaEstimada": 14.87,
  "dataPrevistaColheita": "2026-06-15",
  "confianca": 91.3,
  "status": "CRESCIMENTO_IDEAL"
}
```

**Status possíveis:**

| Status | Biomassa (g/L) |
|---|---|
| `CRESCIMENTO_IDEAL` | ≥ 15.0 |
| `CRESCIMENTO_MODERADO` | ≥ 10.0 |
| `CRESCIMENTO_LENTO` | ≥ 5.0 |
| `ALERTA_BAIXA_BIOMASSA` | < 5.0 |

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "servico": "biomassa-ia",
  "versao": "1.0.0"
}
```

---

## Treinamento do Modelo

### Primeira vez (sem dados do Oracle ainda)

O script gera 500 amostras sintéticas para demonstração:

```bash
python -m training.train
```

Isso cria `artifacts/model.pkl` e `artifacts/scaler.pkl`.

### Com dados reais do Oracle

Edite o final do `training/train.py` para usar `carregar_dados_oracle()` em vez dos dados sintéticos:

```python
# training/train.py — bloco __main__
if __name__ == "__main__":
    df = carregar_dados_oracle()   # substitui os dados sintéticos
    treinar(df)
```

Então rode:
```bash
python -m training.train
```

---

## Re-treinamento Contínuo

O re-treinamento automático é gerenciado pelo `training/scheduler.py` e inicia junto com a API.

Adicione ao `app/main.py` (se ainda não estiver):

```python
from training.scheduler import iniciar_scheduler

@app.on_event("startup")
async def ao_iniciar():
    carregar_modelo()
    iniciar_scheduler()   # ← adicione esta linha
```

**Fluxo do scheduler:**

```
A cada RETRAIN_INTERVAL_HOURS horas
    │
    ├─ Conta registros novos no Oracle desde o último treino
    │
    ├─ Se novos < RETRAIN_MIN_NEW_RECORDS → aguarda a próxima rodada
    │
    ├─ Carrega todos os dados históricos
    ├─ Treina novo Random Forest
    ├─ Compara MAE do novo modelo com o atual
    │
    ├─ Se novo modelo for melhor (ou igual) → hot-swap em memória + salva .pkl
    └─ Se novo modelo for pior → descarta, mantém o atual
```

Os metadados de cada treino (data, MAE, R², quantidade de registros) são salvos em `artifacts/meta.json`.

---

## Banco de Dados Oracle

### Criação das tabelas

Execute no SQL Developer ou SQL*Plus:

```sql
-- Medições dos sensores IoT
CREATE TABLE medicoes_sensor (
    id             NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tanque_id      NUMBER         NOT NULL,
    ph             NUMBER(5, 2),
    temperatura    NUMBER(5, 2),
    turbidez       NUMBER(8, 2),
    luminosidade   NUMBER(10, 2),
    radiacao_par   NUMBER(8, 3),
    data_medicao   DATE           NOT NULL,
    criado_em      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Resultado real das colheitas (ground truth do modelo)
CREATE TABLE colheitas (
    id               NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tanque_id        NUMBER         NOT NULL,
    data_referencia  DATE           NOT NULL,
    biomassa_final   NUMBER(10, 2),
    criado_em        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Previsões geradas pela IA
CREATE TABLE previsoes_biomassa (
    id                     NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tanque_id              NUMBER         NOT NULL,
    biomassa_estimada      NUMBER(10, 2),
    data_prevista_colheita DATE,
    confianca              NUMBER(5, 2),
    status                 VARCHAR2(50),
    criado_em              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para performance nas queries de treinamento
CREATE INDEX idx_medicoes_tanque_data  ON medicoes_sensor (tanque_id, data_medicao);
CREATE INDEX idx_colheitas_tanque_data ON colheitas       (tanque_id, data_referencia);
CREATE INDEX idx_colheitas_criado_em   ON colheitas       (criado_em);
```

### Verificar conexão

```bash
python -c "from app.db.oracle import get_conexao; c = get_conexao(); print('✅ Conectado:', c.version)"
```

---

## Testes

```bash
# Todos os testes
pytest tests/ -v

# Com cobertura de código
pytest tests/ -v --cov=app --cov-report=term-missing

# Um teste específico
pytest tests/test_predictor.py::test_previsao_retorna_saida_valida -v
```

**Testes disponíveis:**

| Teste | O que verifica |
|---|---|
| `test_status_crescimento_ideal` | Classificação de status ≥ 15 g/L |
| `test_status_crescimento_moderado` | Classificação de status ≥ 10 g/L |
| `test_status_crescimento_lento` | Classificação de status ≥ 5 g/L |
| `test_status_alerta` | Classificação de status < 5 g/L |
| `test_previsao_retorna_saida_valida` | Estrutura completa da resposta |
| `test_condicoes_ruins_geram_baixa_biomassa` | Lógica do modelo (sanity check) |
| `test_schemas_valida_ph_invalido` | Validação Pydantic (pH > 14 rejeitado) |

---

## Deploy na OCI

A API é leve — consome ~150–300 MB de RAM com o modelo em memória. Uma VM `VM.Standard.E2.1.Micro` (Always Free) é suficiente para testes e cargas baixas.

### 1. Provisionar a VM

No Console da OCI, crie uma instância:
- Shape: `VM.Standard.E2.1.Micro` (Always Free) ou superior
- Imagem: Oracle Linux 8 ou Ubuntu 22.04
- Abra a porta `8000` no Security List (ou use a porta `80` com proxy)

### 2. Configurar a VM

```bash
# Conecte via SSH
ssh ubuntu@163.176.225.114

# Instale Python 3.11
sudo dnf install python3.11 python3.11-pip git -y   # Oracle Linux
# ou: sudo apt install python3.11 python3.11-venv git -y  # Ubuntu

# Clone o projeto
git clone https://github.com/GS1-2TDSPG-2026/gs1-python-LSTM.git
cd gs1-python-LSTM

# Ambiente virtual e dependências
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install apscheduler==3.10.4

# Configure o .env com as credenciais do Oracle
cp .env.example .env
nano .env

# Treine o modelo inicial
python -m training.train
```

### 3. Rodar como serviço (systemd)

```bash
sudo nano /etc/systemd/system/biomassa-ia.service
```

Cole o conteúdo abaixo (ajuste o caminho se necessário):

```ini
[Unit]
Description=Biomassa IA — Motor Preditivo
After=network.target

[Service]
Type=simple
User=opc
WorkingDirectory=/home/opc/biomassa-ia
EnvironmentFile=/home/opc/biomassa-ia/.env
ExecStart=/home/opc/biomassa-ia/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable biomassa-ia
sudo systemctl start biomassa-ia

# Verificar status
sudo systemctl status biomassa-ia

# Ver logs em tempo real
sudo journalctl -u biomassa-ia -f
```

### 4. Abrir a porta no firewall da VM

```bash
# Oracle Linux (firewalld)
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload

# Ubuntu (ufw)
sudo ufw allow 8000/tcp
```

Lembre-se também de liberar a porta `8000` no **Security List** da subnet no Console da OCI.

### Consumo de recursos estimado (Always Free)

| Recurso | Em idle | Durante re-treino |
|---|---|---|
| RAM | ~180 MB | ~280 MB |
| CPU | < 1% | ~80% por ~10–30s |
| Disco | < 50 MB | < 50 MB |

O re-treinamento dura apenas alguns segundos e roda em background — a API continua respondendo normalmente durante esse período.

---

## Estrutura do Projeto

```
biomassa-ia/
├── app/
│   ├── api/
│   │   └── routes.py          # Endpoints FastAPI
│   ├── data/
│   │   └── preprocessor.py    # Normalização e classificação de status
│   ├── db/
│   │   └── oracle.py          # Leitura e escrita no Oracle
│   ├── models/
│   │   ├── predictor.py       # Carrega modelo e executa previsão
│   │   └── schemas.py         # Contratos Pydantic (request/response)
│   └── main.py                # Ponto de entrada da aplicação
├── artifacts/
│   ├── model.pkl              # Modelo treinado (gerado por train.py)
│   ├── scaler.pkl             # Normalizador (gerado por train.py)
│   └── meta.json              # Metadados do último treino
├── tests/
│   └── test_predictor.py      # Testes automatizados
├── training/
│   ├── scheduler.py           # Re-treinamento contínuo (APScheduler)
│   └── train.py               # Script de treinamento manual
├── .env.example               # Template de variáveis de ambiente
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Licença

MIT — veja o arquivo [LICENSE](LICENSE) para detalhes.
