# 🧪 Testes Manuais com Pytest (Python 3.13)

Este documento descreve como executar os testes automatizados do projeto manualmente, utilizando **Python 3.13** e também via **Docker**.

---

## ⚠️ Observação sobre Python 3.13

O projeto está sendo executado com:

```bash
python --version

Exemplo:

Python 3.13.x

🔴 Atenção:
Algumas bibliotecas podem ter suporte parcial no Python 3.13:

TensorFlow
PyTorch (dependendo da versão)
MLflow

👉 Apesar disso, os testes funcionam normalmente utilizando CPU.

📁 Estrutura de Testes

tests/
├── conftest.py
├── test_api.py
├── test_ingest.py
├── test_baseline.py
├── test_train.py
├── test_plot_metrics.py
├── test_feature_engineering.py

⚙️ Pré-requisitos

Execução local (sem Docker)
Python 3.13

Ambiente virtual ativo (recomendado)

Execução via Docker

Docker Desktop instalado
Docker em execução
(WSL integrado, se estiver usando Windows)

🧪 Execução LOCAL (sem Docker)

Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate

Instalar dependências
pip install -e ".[test]"

Rodar testes
pytest

🐳 Execução via Docker (RECOMENDADO)

O projeto possui ambiente isolado para testes utilizando Dockerfile.test.

🔧 Build da imagem
docker compose build test

ou:

docker build -f Dockerfile.test -t datathon-mlet-test .

▶️ Executar testes
Docker Compose
docker compose run --rm test
Docker direto
docker run --rm datathon-mlet-test

📊 Saída esperada
collected XX items

tests/test_api.py::test_health_returns_200 PASSED
tests/test_ingest.py::test_ingest_success PASSED
tests/test_baseline.py::test_avaliar_modelo PASSED
tests/test_feature_engineering.py::test_run_pipeline PASSED
...

🧪 Testes de Feature Engineering

O arquivo:

tests/test_feature_engineering.py

valida:

criação das features do EDA
tratamento de outliers
remoção de correlação
cálculo de VIF
seleção automática de features
aplicação de scaler
aplicação de PCA
geração de relatórios
execução completa do pipeline

👉 Esses testes garantem que o pipeline funcione mesmo com mudanças nas features.

🧠 Observações importantes

✔ 1. Imports funcionando automaticamente

O projeto utiliza configuração no pyproject.toml:

pythonpath = ["."]

👉 Permite que o pytest encontre:

src/
data/
app/

✔ 2. Fixtures (conftest.py)

Os testes utilizam dados sintéticos:

simulação de dados de ações
arquivos CSV temporários
cenários inválidos

✔ 3. Testes com Mock

Uso de monkeypatch para:

simular yfinance.download
evitar chamadas externas
garantir execução offline

✔ 4. Execução em CPU

Mensagem esperada:

CUDA initialization...
GPU will not be used

✔ 5. Docker vs Local

Execução	Quando usar
Local (pytest)	desenvolvimento rápido
Docker	ambiente isolado / CI

✔ 6. Docker Desktop

Para execução via Docker:

Docker Desktop deve estar instalado
serviço deve estar em execução
WSL integrado (Windows)

🧪 Cobertura de Testes
pytest --cov=src --cov=data

ou:

pytest --cov=src --cov=data --cov-report=term-missing

🚀 Boas práticas aplicadas

✔ Testes isolados
✔ Uso de dados sintéticos
✔ Mock de APIs externas
✔ Testes de Machine Learning
✔ Testes de Feature Engineering
✔ Testes de geração de gráficos
✔ Validação de erros

📌 Execução resumida

Local
pip install -e ".[test]"
pytest
Docker
docker compose build test
docker compose run --rm test