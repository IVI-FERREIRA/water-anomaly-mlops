# 💧 Water Anomaly Detection — MLOps Project

Projeto de **detecção de anomalias na qualidade da água** utilizando Machine Learning, com pipeline organizado, API de inferência e deploy via Docker.

---

## 🎯 Objetivo

Detectar **comportamentos anômalos** em dados de sensores de qualidade da água, aplicando boas práticas de **MLOps**, desde o preparo dos dados até a disponibilização do modelo em produção.

---

## 🧠 Visão Geral da Solução

- Os dados passam por um processo de **limpeza e preparação**
- Um modelo de **Isolation Forest** aprende o padrão de normalidade
- O modelo treinado é exposto através de uma **API FastAPI**
- A aplicação pode ser executada localmente ou via **Docker**

---

## 🏗️ Arquitetura
<img width="1723" height="495" alt="image" src="https://github.com/user-attachments/assets/894cb797-3068-4cca-ade5-6bfaf5234d2f" />


## 📁 Estrutura do Repositório
```text
data/
├── sample/           # Dataset de exemplo (para testes)
└── processed/        # Dados tratados (gerados no pipeline)

docker/
└── Dockerfile        # Container da aplicação

models/
└── model.joblib      # Modelo treinado (ignorado no Git)

src/
├── api/
│   └── main.py       # API FastAPI
├── data_prep.py      # Preparação dos dados
├── train.py          # Treinamento do modelo
└── infer.py          # Inferência local (opcional)

.gitignore
requirements.txt
README.md
```
## 🚀 COMO RODAR LOCALMENTE
Clone o projeto para sua máquina local com o comando:  git clone https://github.com/IVI-FERREIRA/water-anomaly-mlops.git

### 1️⃣ Criar ambiente virtual
```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
### 2️⃣ Prepara os dados 
python src/data_prep.py

### 3️⃣ Treinar o modelo
python src/train.py

### 4️⃣ Rodar a API
uvicorn src.api.main:app --reload


Acesse:
http://127.0.0.1:8000/docs



## 📡 Endpoint da API
POST /predict

Recebe um JSON com todas as colunas do modelo e retorna:

{ "result": "NORMAL" }


ou

{ "result": "ANOMALIA" }



## 🐳 Rodar com Docker
```text
-Build da imagem
docker build -t water-anomaly-api -f docker/Dockerfile .

-Executar container
docker run -p 8000:8000 water-anomaly-api


Acesse:
http://127.0.0.1:8000/docs
```
## 📊 Tecnologias Utilizadas

-Python

-Pandas

-Scikit-learn

-FastAPI

-Docker

🔧 Possíveis Melhorias

Versionamento de modelos com MLflow

Monitoramento de data/model drift

Pipeline de CI/CD

Validação de entrada com Pydantic

Deploy em AWS Lambda ou Kubernetes
