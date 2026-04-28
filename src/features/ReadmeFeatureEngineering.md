# 🧩 Feature Engineering Pipeline

Este documento descreve a etapa de **Feature Engineering** do projeto *Datathon MLET*, responsável por transformar dados brutos do mercado financeiro em um dataset estruturado e pronto para Machine Learning.

---

# ⚙️ Ambiente

## 🐍 Versão do Python

```bash
Python 3.13.x

⚠️ Algumas bibliotecas possuem suporte parcial ao Python 3.13, porém o pipeline funciona corretamente em CPU.

📦 Principais Bibliotecas
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
statsmodels
joblib
yfinance>=0.2.40
pytest
pytest-cov

🧪 Ambiente de Testes (pyenv)

Instalação do Python
pyenv install 3.13.0

Criar ambiente virtual
pyenv virtualenv 3.13.0 datathon-env

Ativar no projeto
cd ~/datathon-mlet
pyenv local datathon-env

Verificar
python --version

Instalar dependências
pip install -e .[dev]

Executar testes
PYTHONPATH=. pytest -v tests/

Cobertura de testes
pip install pytest-cov
PYTHONPATH=. pytest --cov=src --cov=data -v

🎯 Objetivo

Converter dados simples de mercado em informações mais ricas para modelos preditivos.

👉 Antes:

preços brutos

👉 Depois:

indicadores inteligentes

📥 Fonte dos Dados
data/raw/stock_data.csv

Contém:

Open
High
Low
Close
Volume

⚙️ Pipeline de Feature Engineering

1. 📦 Ingestão
df = load_data()

2. 🧹 Limpeza
Conversão numérica
Remoção de inválidos
Tratamento de NaN

3. 🧠 Features do EDA
Feature	Descrição
retorno	variação percentual
volatilidade_20	volatilidade
mm_20	média móvel 20
mm_50	média móvel 50
amplitude	High - Low
variacao_dia	Close - Open

4. 🎯 Target
target_next_close

5. ⚠️ Outliers

Winsorização:

1% - 99%
6. 🔗 Correlação

Remove features redundantes (>0.90)

📄 Output:

reports/correlation_matrix.csv
reports/removed_by_correlation.csv

7. 📊 VIF

Remove multicolinearidade:

VIF > 10

📄 Output:

reports/vif_report.csv
reports/removed_by_vif.csv

8. 🤖 Seleção de Features
SelectKBest(f_regression)

📄 Output:

reports/selected_features.csv

9. 📏 Scaling
RobustScaler()

📄 Output:

models/scaler_features.joblib

10. 🔥 PCA
Redução de dimensionalidade
Mantém 95% da variância

📄 Output:

models/pca_features.joblib
reports/pca_explained_variance.csv

11. 💾 Output Final
data/raw/stock_features.csv

📊 Estrutura Final
stock_features.csv
│
├── retorno
├── volatilidade_20
├── mm_20
├── mm_50
├── amplitude
├── variacao_dia
│
├── *_scaled
│
├── pca_1
├── pca_2
│
└── target_next_close

🚀 Execução
python src/features/feature_engineering.py

📊 Relatórios
reports/

Contém:

correlação
VIF
seleção de features
PCA

🧠 Inteligência do Pipeline

O pipeline é dinâmico:

✔ adapta mudanças de features
✔ recalcula automaticamente
✔ mantém consistência

🔥 Benefícios

✔ Redução de overfitting
✔ Dados limpos
✔ Pipeline reprodutível
✔ Melhor performance

⚠️ Observações
PCA reduz interpretabilidade
Scaling deve ser separado em produção
Compatível com DVC

🔗 Integração
ingest → feature_engineering → train → mlflow

🏁 Conclusão

Esta etapa transforma dados brutos em um dataset otimizado para Machine Learning, aumentando a qualidade das previsões.