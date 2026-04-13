# Projeto MLOps — Census Income Classification API

> Guia de execução em estilo **sprint completa**, organizado para uso no **Obsidian** e para servir como referência de implementação, documentação e entrega do projeto.

---

## 1. Visão geral do projeto

Neste projeto, o objetivo é desenvolver uma solução de **Machine Learning em produção** a partir do dataset **Census Income (Adult Dataset)**, criando um fluxo que vai de **treinamento do modelo** até **deploy de uma API com FastAPI**, com **testes automatizados**, **validação por slices** e **CI/CD com GitHub Actions**.

O problema de negócio é um problema clássico de **classificação binária**:

- prever se a renda anual de uma pessoa **excede 50 mil dólares por ano**;
- target:
  - `>50K`
  - `<=50K`

Esse projeto é importante porque cobre, em uma única entrega, várias competências de nível profissional:

- preparação e limpeza de dados tabulares;
- treinamento de modelo de classificação;
- persistência de artefatos de ML;
- avaliação global e por subgrupos;
- documentação do modelo com **Model Card**;
- serving com **FastAPI**;
- testes unitários do pipeline e da API;
- integração contínua e entrega contínua;
- deploy em plataforma cloud.

---

## 2. Objetivo da sprint

Ao final da sprint, você deverá ter:

- um repositório GitHub organizado;
- um pipeline de treinamento funcional;
- artefatos do modelo salvos;
- testes do módulo de ML;
- testes da API;
- um arquivo `slice_output.txt` com métricas por slices;
- um `model_card.md` preenchido;
- uma API FastAPI com documentação automática e exemplo de payload;
- CI com GitHub Actions rodando `pytest` e `flake8`;
- deploy automático em uma plataforma cloud;
- script que faz `POST` na API em produção;
- screenshots exigidos pela rubrica.

---

## 3. Dataset

## 3.1 Nome do dataset

**Census Income / Adult Dataset**

## 3.2 Tarefa

**Classificação binária**

## 3.3 Variável alvo

- `income`
  - `>50K`
  - `<=50K`

## 3.4 Principais características do dataset

- dados tabulares;
- mistura de variáveis:
  - categóricas;
  - inteiras;
  - binárias;
- possui valores ausentes em algumas colunas categóricas;
- contém nomes de colunas com hífen, por exemplo:
  - `education-num`
  - `marital-status`
  - `capital-gain`
  - `capital-loss`
  - `hours-per-week`
  - `native-country`

## 3.5 Variáveis mais relevantes

### Numéricas
- `age`
- `fnlwgt`
- `education-num`
- `capital-gain`
- `capital-loss`
- `hours-per-week`

### Categóricas
- `workclass`
- `education`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native-country`

### Target
- `income`

## 3.6 Observações práticas

O enunciado informa que o dado vem “messy” e sugere remover espaços para limpeza inicial. Em uma implementação mais madura, isso pode ser tratado programaticamente com pandas, por exemplo:

- remoção de espaços em volta de strings;
- padronização de categorias;
- tratamento de `?` ou missing values;
- separação explícita entre preprocessamento de treino e inferência.

---

## 4. O que a rubrica realmente avalia

Este projeto **não** avalia apenas se “o modelo funciona”. Ele avalia uma solução de ML como produto de engenharia.

Os eixos de avaliação são:

1. **Git + GitHub Actions**
2. **Treinamento e persistência do modelo**
3. **Testes unitários**
4. **Validação em slices**
5. **Model Card**
6. **API REST com FastAPI**
7. **Testes da API**
8. **Deploy em nuvem**
9. **Script consumindo a API live**
10. **Evidências visuais exigidas pela rubrica**

---

## 5. Stack recomendada para o projeto

Como você vai trabalhar no seu computador pessoal e deseja usar **uv** como gerenciador de dependências, a stack recomendada pode ser:

- **Python 3.13**
- **uv** para ambiente e dependências
- **pandas**
- **numpy**
- **scikit-learn**
- **FastAPI**
- **uvicorn**
- **pydantic**
- **pytest**
- **httpx**
- **requests**
- **flake8**
- **Git**
- **GitHub Actions**
- opcional:
  - **DVC**
  - **Render** ou **Heroku**

---

## 6. Estrutura sugerida do projeto

Uma estrutura simples, clara e adequada para esse projeto:

```text
census-income-mlops/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── predict.py
├── data/
│   ├── census.csv
│   └── census_clean.csv
├── model/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── slices.py
├── artifacts/
│   ├── model.pkl
│   ├── encoder.pkl
│   └── lb.pkl
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   └── test_api.py
├── scripts/
│   ├── train_model.py
│   ├── compute_slices.py
│   └── post_live_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── screenshots/
│   ├── example.png
│   ├── continuous_integration.png
│   ├── continuous_deloyment.png
│   ├── live_get.png
│   └── live_post.png
├── slice_output.txt
├── model_card.md
├── README.md
├── pyproject.toml
└── .gitignore
```

---

## 7. Setup local com uv

## 7.1 Pré-requisitos

Instale previamente:

- Python 3.13
- Git
- uv

## 7.2 Criar o projeto

```bash
mkdir census-income-mlops
cd census-income-mlops
git init
```

## 7.3 Inicializar o ambiente com uv

```bash
uv venv --python 3.13
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 7.4 Criar `pyproject.toml`

Você pode iniciar com:

```bash
uv init
```

ou criar manualmente um `pyproject.toml`.

## 7.5 Instalar dependências

```bash
uv add fastapi "uvicorn[standard]" pydantic pandas numpy scikit-learn requests httpx pytest matplotlib seaborn jupyter ipykernel
uv add --dev flake8
```

Se quiser manter alinhado com o que foi estudado no material:

```bash
uv add fastapi==0.117.1 "uvicorn[standard]==0.36.0" pydantic==2.11.9 pandas==2.3.2 numpy==2.3.3 scikit-learn==1.7.2 requests==2.32.5 httpx==0.28.1 pytest==8.4.2 matplotlib==3.10.6 seaborn==0.13.2 jupyter==1.1.1 ipykernel==6.30.1
uv add --dev flake8
```

## 7.6 Exportar requirements se necessário

Caso você precise de `requirements.txt` para deploy:

```bash
uv export --format requirements-txt > requirements.txt
```

---

## 8. Sprint macro — visão de execução

A sprint pode ser organizada em 8 blocos:

1. **Fundação do projeto**
2. **Entendimento e limpeza dos dados**
3. **Pipeline de treino**
4. **Avaliação e validação por slices**
5. **Model Card**
6. **API com FastAPI**
7. **Testes + CI**
8. **Deploy + evidências finais**

---

## 9. Sprint 1 — Fundação do projeto

## 9.1 Objetivo
Preparar o ambiente, estrutura do repositório e base de desenvolvimento.

## 9.2 Entregáveis
- repositório inicializado;
- ambiente local funcional;
- estrutura de pastas criada;
- dependências instaladas;
- primeiro commit.

## 9.3 Passos

### Criar estrutura de diretórios

```bash
mkdir -p app model tests scripts data artifacts screenshots .github/workflows
touch README.md model_card.md slice_output.txt
```

### Criar `.gitignore`

Sugestão:

```gitignore
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
.coverage
htmlcov/
.DS_Store
```

### Fazer commit inicial

```bash
git add .
git commit -m "chore: initialize project structure"
```

---

## 10. Sprint 2 — Entendimento e limpeza dos dados

## 10.1 Objetivo
Ler o dataset, entender colunas, target, tipos e limpar os dados.

## 10.2 Perguntas que você deve responder
- qual é a variável alvo?
- quais features são categóricas?
- quais são numéricas?
- existem missing values?
- como os missing values aparecem?
- existem espaços extras nas categorias?
- existem colunas com nomes problemáticos para uso em Python?
- como será o preprocessamento para treino e inferência?

## 10.3 Tarefas

- carregar `census.csv` com pandas;
- inspecionar:
  - shape
  - `dtypes`
  - `head`
  - classes da target
  - missing values
- produzir uma versão limpa, por exemplo:
  - remover espaços em volta de strings;
  - padronizar target;
  - tratar `?` como missing quando necessário;
  - salvar `data/census_clean.csv`.

## 10.4 Dica de implementação

Como o dataset contém colunas categóricas, o pipeline de preprocessamento pode separar:

- `cat_features`
- `num_features`

Exemplo conceitual:

```python
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

num_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
```

---

## 11. Sprint 3 — Pipeline de treino

## 11.1 Objetivo
Treinar um modelo de classificação e salvar todos os artefatos necessários para inferência.

## 11.2 O que a rubrica espera minimamente
Você deve ter funções para:

- treinar o modelo;
- salvar o modelo;
- carregar o modelo;
- fazer inferência;
- computar métricas de classificação.

## 11.3 Arquitetura recomendada

### `model/data.py`
Responsável por:
- carregar dados;
- separar `X` e `y`;
- eventualmente fazer limpeza simples.

### `model/features.py`
Responsável por:
- identificar colunas categóricas;
- montar preprocessamento;
- encoder.

### `model/train.py`
Responsável por:
- split treino/teste;
- ajuste do pipeline ou modelo;
- persistência dos artefatos.

### `model/inference.py`
Responsável por:
- carregar artefatos;
- receber novos dados;
- transformar;
- prever.

### `model/evaluate.py`
Responsável por:
- precision;
- recall;
- fbeta;
- outras métricas úteis.

## 11.4 Modelo recomendado

Para este projeto, uma escolha clássica e robusta é:

- `RandomForestClassifier`

Alternativas:
- Logistic Regression
- XGBoost, se quiser algo mais forte, mas não é necessário para o escopo.

## 11.5 Estratégia prática recomendada

### Pré-processamento
- `OneHotEncoder(handle_unknown="ignore")` para categóricas;
- passthrough ou scaler para numéricas.

### Split
- `train_test_split(test_size=0.20, random_state=42, stratify=y)`

### Persistência
Salvar:
- modelo treinado;
- encoder/preprocessor;
- label binarizer, se usado.

## 11.6 Script de treino

Crie um script reprodutível:

`scripts/train_model.py`

Esse script deve:
1. carregar dados limpos;
2. separar treino/teste;
3. preprocessar;
4. treinar o modelo;
5. avaliar;
6. salvar artefatos.

### Exemplo de execução

```bash
python scripts/train_model.py
```

---

## 12. Sprint 4 — Avaliação e validação por slices

## 12.1 Objetivo
Medir a performance do modelo não apenas no agregado, mas também em subgrupos dos dados.

## 12.2 Por que isso é importante
Um modelo pode ter boa performance média, mas piorar fortemente em segmentos específicos, por exemplo:

- mulheres vs homens;
- diferentes níveis de educação;
- diferentes ocupações;
- diferentes grupos raciais;
- diferentes países de origem.

A validação por slices ajuda a detectar:

- instabilidade de performance;
- risco de viés;
- fragilidade operacional.

## 12.3 O que a rubrica exige
- criar uma função que compute métricas para cada valor único de uma feature categórica;
- gerar saída para todos os valores;
- salvar em `slice_output.txt`.

## 12.4 Estratégia sugerida

Exemplo:
- escolher cada feature categórica;
- para cada valor único:
  - filtrar o subconjunto;
  - rodar inferência;
  - calcular precision, recall, fbeta;
  - registrar resultado.

## 12.5 Formato sugerido de saída

```text
Feature: education
Slice: Bachelors
Precision: 0.72
Recall: 0.65
Fbeta: 0.68

Feature: education
Slice: HS-grad
Precision: 0.61
Recall: 0.53
Fbeta: 0.57
```

## 12.6 Script recomendado

`scripts/compute_slices.py`

### Execução
```bash
python scripts/compute_slices.py
```

### Saída obrigatória
```text
slice_output.txt
```

---

## 13. Sprint 5 — Model Card

## 13.1 Objetivo
Documentar o modelo de forma clara, estruturada e profissional.

## 13.2 Estrutura exigida

O template estudado contém:

- Model Details
- Intended Use
- Training Data
- Evaluation Data
- Metrics
- Ethical Considerations
- Caveats and Recommendations

## 13.3 Como preencher cada seção

### Model Details
Descreva:
- nome do modelo;
- algoritmo;
- versão;
- data;
- autor;
- framework utilizado.

### Intended Use
Explique:
- o objetivo do modelo;
- qual problema ele resolve;
- quem deveria usar;
- quem não deveria usar;
- contexto de uso permitido.

### Training Data
Descreva:
- origem do dataset;
- período e contexto;
- features utilizadas;
- target;
- limpeza aplicada;
- limitações da base.

### Evaluation Data
Explique:
- como foi feito o split;
- tamanho do conjunto de teste;
- se houve estratificação;
- se a avaliação representa bem o caso real.

### Metrics
Inclua:
- métricas usadas;
- desempenho do modelo nessas métricas;
- idealmente também uma breve interpretação.

### Ethical Considerations
Discuta:
- risco de viés por atributos demográficos;
- risco de uso indevido;
- possível impacto em decisões humanas;
- necessidade de supervisão.

### Caveats and Recommendations
Inclua:
- limitações do dataset;
- risco de drift;
- necessidade de monitoramento;
- recomendações de melhoria.

## 13.4 Nome sugerido do arquivo
```text
model_card.md
```

---

## 14. Sprint 6 — API com FastAPI

## 14.1 Objetivo
Expor o modelo como um serviço HTTP para inferência.

## 14.2 Requisitos da rubrica
A API deve ter:

- `GET /` na raiz com mensagem de boas-vindas;
- `POST` em outro endpoint para inferência;
- type hints;
- modelo Pydantic para o corpo;
- exemplo de payload visível no Swagger.

## 14.3 Desenho recomendado

### `app/schemas.py`
Definir o modelo Pydantic de entrada.

### `app/predict.py`
Concentrar lógica de predição.

### `app/main.py`
Subir a aplicação FastAPI e definir rotas.

## 14.4 Ponto técnico crítico: colunas com hífen

Como Python não aceita nomes de atributos com hífen, use **aliases** no Pydantic.

Exemplo conceitual:

```python
from pydantic import BaseModel, Field, ConfigDict

class CensusInput(BaseModel):
    age: int
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    education: str
    education_num: int = Field(alias="education-num")
    fnlwgt: int
    hours_per_week: int = Field(alias="hours-per-week")
    marital_status: str = Field(alias="marital-status")
    native_country: str = Field(alias="native-country")
    occupation: str
    race: str
    relationship: str
    sex: str
    workclass: str

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 45,
                "capital-gain": 2174,
                "capital-loss": 0,
                "education": "Bachelors",
                "education-num": 13,
                "fnlwgt": 233,
                "hours-per-week": 60,
                "marital-status": "Never-married",
                "native-country": "Cuba",
                "occupation": "Prof-specialty",
                "race": "Black",
                "relationship": "Wife",
                "sex": "Female",
                "workclass": "State-gov"
            }
        }
    )
```

## 14.5 Endpoints sugeridos

### GET `/`
Retorna algo como:

```json
{"message": "Welcome to the Census Income Prediction API"}
```

### POST `/model/`
Recebe o payload e retorna algo como:

```json
{
  "prediction": ">50K"
}
```

ou

```json
{
  "prediction": "<=50K"
}
```

## 14.6 Rodar localmente

```bash
uvicorn app.main:app --reload
```

Swagger:
```text
http://127.0.0.1:8000/docs
```

---

## 15. Sprint 7 — Testes

## 15.1 Objetivo
Garantir qualidade mínima do código e da API.

## 15.2 O que a rubrica pede
No final, deve haver pelo menos **seis testes**:

- pelo menos 3 testes do código de ML;
- pelo menos 3 testes da API.

## 15.3 Testes do módulo de modelo

Você pode testar, por exemplo:

1. se a função de treino retorna um objeto do tipo esperado;
2. se a função de inferência retorna array/lista com tamanho esperado;
3. se a função de métricas retorna floats ou estrutura correta.

## 15.4 Testes da API

A rubrica exige:

1. **um teste para o GET**
   - deve validar status code;
   - deve validar conteúdo retornado.

2. **um teste para cada possível saída do modelo**
   - um payload que gere `<=50K`;
   - um payload que gere `>50K`.

## 15.5 Arquivo sugerido
```text
tests/test_api.py
tests/test_model.py
```

## 15.6 Executar testes

```bash
pytest -q
```

---

## 16. Sprint 8 — Sanity check

Se você tiver o script `sanitycheck.py` do starter code, rode:

```bash
python sanitycheck.py
```

Ele verifica problemas comuns nos testes de API, especialmente:

- ausência de teste para cada saída;
- testes superficiais demais;
- problemas no GET ou POST.

Mesmo passando no sanity check, ainda vale revisar manualmente a rubrica.

---

## 17. Integração contínua com GitHub Actions

## 17.1 Objetivo
Automatizar validação do projeto a cada push.

## 17.2 Requisito mínimo
Rodar:

- `pytest`
- `flake8`

em push para `main` ou `master`.

## 17.3 Workflow sugerido

Arquivo:
```text
.github/workflows/ci.yml
```

Exemplo:

```yaml
name: CI

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: |
          uv venv
          uv sync || true
          source .venv/bin/activate
          uv pip install -r requirements.txt

      - name: Run flake8
        run: |
          source .venv/bin/activate
          flake8 .

      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest
```

## 17.4 Observação prática
Se você usar `pyproject.toml` de forma madura, adapte o workflow para `uv sync` corretamente. Se o deploy exigir `requirements.txt`, mantenha também esse arquivo exportado.

---

## 18. Deploy em nuvem

## 18.1 Objetivo
Subir a API para uma plataforma cloud com deploy automático.

## 18.2 Plataformas possíveis
- **Render**
- **Heroku**

O material menciona Heroku, mas você pode decidir conforme conveniência e custo.

## 18.3 Requisito da rubrica
- o app deve ser implantado a partir do GitHub;
- o deploy automático deve ocorrer **somente se o CI passar**.

## 18.4 O que validar antes do deploy
- `requirements.txt` atualizado;
- comando de start definido;
- paths dos artefatos corretos;
- a API consegue carregar modelo em ambiente de produção.

## 18.5 Comando de start típico
```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

---

## 19. DVC — quando usar e como encaixar

O material estudado menciona cenários com **DVC**, inclusive:

- versionamento de dados e modelos;
- uso de bucket remoto;
- instruções específicas para Heroku;
- `dvc pull` ao iniciar o app em produção.

## 19.1 Quando vale usar aqui
Use DVC se você quiser transformar este projeto em uma versão mais forte de portfólio, com:

- versionamento explícito do dataset;
- versionamento de artefatos do modelo;
- integração com remoto S3 ou similar.

## 19.2 Quando não usar
Se sua prioridade for concluir primeiro a rubrica mínima com segurança, você pode começar sem DVC e adicionar isso depois como incremento.

## 19.3 Estratégia recomendada
- **primeira passagem:** fechar rubrica mínima;
- **segunda passagem:** adicionar DVC como refinamento de portfólio.

---

## 20. Script para consultar a API em produção

## 20.1 Objetivo
Escrever um script com `requests` que envie um `POST` para a API live e imprima:

- status code;
- resposta da inferência.

## 20.2 Arquivo sugerido
```text
scripts/post_live_api.py
```

## 20.3 Estrutura conceitual

```python
import requests

url = "https://SEU-APP/model/"

payload = {
    "age": 45,
    "capital-gain": 2174,
    "capital-loss": 0,
    "education": "Bachelors",
    "education-num": 13,
    "fnlwgt": 233,
    "hours-per-week": 60,
    "marital-status": "Never-married",
    "native-country": "Cuba",
    "occupation": "Prof-specialty",
    "race": "Black",
    "relationship": "Wife",
    "sex": "Female",
    "workclass": "State-gov"
}

response = requests.post(url, json=payload, timeout=30)

print("Status code:", response.status_code)
print("Response:", response.json())
```

---

## 21. Screenshots obrigatórias

A rubrica exige evidências visuais com nomes específicos.

## 21.1 `example.png`
Screenshot do Swagger mostrando o exemplo do payload no endpoint POST.

## 21.2 `continuous_integration.png`
Screenshot do CI passando no GitHub Actions  
ou, alternativamente, link do repo no README.

## 21.3 `continuous_deloyment.png`
Screenshot mostrando o deploy automático habilitado.  
> Observação: a rubrica apresenta esse nome com `deloyment`. Siga exatamente esse nome.

## 21.4 `live_get.png`
Screenshot do navegador acessando a URL live no endpoint raiz (`GET /`).

## 21.5 `live_post.png`
Screenshot do script ou resultado do POST na API em produção.

---

## 22. Checklist de conclusão da rubrica

## 22.1 Git + CI
- [ ] repositório no GitHub criado
- [ ] commits feitos ao longo do desenvolvimento
- [ ] GitHub Actions configurado
- [ ] `pytest` roda no CI
- [ ] `flake8` roda no CI
- [ ] CI passa sem erros

## 22.2 Modelo
- [ ] dados carregados corretamente
- [ ] dados limpos
- [ ] split treino/teste implementado
- [ ] modelo treinado
- [ ] artefatos salvos
- [ ] script de treino criado

## 22.3 Testes do modelo
- [ ] pelo menos 3 testes implementados
- [ ] tipos e saídas básicas validados

## 22.4 Slices
- [ ] função de avaliação por slices implementada
- [ ] saída salva em `slice_output.txt`

## 22.5 Model Card
- [ ] todas as seções preenchidas
- [ ] métricas incluídas
- [ ] texto em frases completas

## 22.6 API
- [ ] `GET /` implementado
- [ ] `POST` implementado
- [ ] Pydantic model criado
- [ ] example visível no Swagger
- [ ] type hints presentes

## 22.7 Testes da API
- [ ] teste do GET criado
- [ ] teste para saída `<=50K`
- [ ] teste para saída `>50K`
- [ ] sanity check executado, se disponível

## 22.8 Deploy
- [ ] app em produção
- [ ] deploy automático configurado
- [ ] script de POST live criado

## 22.9 Evidências
- [ ] `example.png`
- [ ] `continuous_integration.png`
- [ ] `continuous_deloyment.png`
- [ ] `live_get.png`
- [ ] `live_post.png`

---

## 23. Ordem recomendada de desenvolvimento

Para reduzir retrabalho, siga esta ordem:

1. estruturar projeto;
2. configurar ambiente com uv;
3. limpar dataset;
4. implementar pipeline de treino;
5. salvar artefatos;
6. implementar métricas;
7. implementar slices;
8. escrever testes do modelo;
9. construir API;
10. escrever testes da API;
11. configurar CI;
12. gerar model card;
13. testar localmente ponta a ponta;
14. fazer deploy;
15. rodar script live;
16. capturar screenshots;
17. revisar rubrica;
18. atualizar README final.

---

## 24. Estratégia de senioridade para este projeto

Como cientista de dados sênior, a recomendação não é apenas “fazer funcionar”, mas deixar a solução com narrativa técnica consistente.

## 24.1 O que demonstra maturidade
- separar responsabilidades por módulo;
- tornar treino e inferência reproduzíveis;
- evitar lógica duplicada entre treino e API;
- documentar limitações do modelo;
- avaliar slices para discutir fairness;
- escrever testes de comportamento, não apenas de existência;
- deixar claro no README como executar tudo;
- manter nomenclatura consistente entre dataset, API e artefatos.

## 24.2 O que evitar
- lógica de preprocessamento espalhada em vários arquivos;
- artefatos carregados com path frágil;
- endpoint POST sem schema claro;
- testes que só validam status 200;
- model card genérico sem métricas;
- screenshots tiradas sem contexto visível.

---

## 25. Plano de execução em formato de grande sprint

## Sprint Day 1 — Fundação
- criar repositório;
- configurar uv;
- instalar dependências;
- organizar estrutura;
- fazer primeiro commit.

## Sprint Day 2 — Dados
- explorar dataset;
- limpar dataset;
- salvar versão limpa;
- documentar colunas e target.

## Sprint Day 3 — Modelo
- implementar preprocessamento;
- treinar modelo;
- salvar artefatos;
- medir métricas iniciais.

## Sprint Day 4 — Avaliação robusta
- criar função de slices;
- gerar `slice_output.txt`;
- interpretar resultados;
- iniciar model card.

## Sprint Day 5 — API
- criar schemas;
- criar GET e POST;
- configurar Swagger com example;
- testar manualmente local.

## Sprint Day 6 — Testes
- escrever testes do modelo;
- escrever testes da API;
- rodar `pytest`;
- rodar `flake8`;
- ajustar falhas.

## Sprint Day 7 — CI + Deploy
- configurar GitHub Actions;
- validar pipeline;
- fazer deploy;
- rodar script live;
- capturar screenshots;
- revisar checklist final.

---

## 26. Entregáveis finais esperados

Ao final, seu projeto deve conter pelo menos:

- código do pipeline de ML;
- artefatos do modelo;
- API FastAPI;
- testes do modelo;
- testes da API;
- `slice_output.txt`;
- `model_card.md`;
- `README.md`;
- workflow de GitHub Actions;
- script de POST live;
- screenshots da rubrica.

---

## 27. Próximo passo recomendado

A melhor forma de começar agora é esta:

1. criar a estrutura do projeto;
2. configurar `uv`;
3. carregar e limpar o dataset;
4. implementar `scripts/train_model.py`;
5. só depois passar para API.

Essa ordem reduz bastante o risco de retrabalho.

---

## 28. Template de tarefas iniciais

Copie esta lista para sua primeira sessão de trabalho:

- [ ] criar pasta do projeto
- [ ] inicializar git
- [ ] criar ambiente com uv
- [ ] instalar dependências
- [ ] criar estrutura de diretórios
- [ ] adicionar dataset em `data/`
- [ ] explorar dataset com pandas
- [ ] limpar dados e salvar `census_clean.csv`
- [ ] implementar pipeline de treino
- [ ] salvar artefatos
- [ ] implementar métricas
- [ ] implementar slices
- [ ] criar `slice_output.txt`
- [ ] escrever `model_card.md`
- [ ] criar FastAPI app
- [ ] adicionar example no Pydantic schema
- [ ] criar testes do modelo
- [ ] criar testes da API
- [ ] configurar GitHub Actions
- [ ] fazer deploy
- [ ] capturar screenshots
- [ ] revisar rubrica final

---

## 29. Conclusão

Este projeto é excelente para consolidar uma transição de:

**modelo em notebook** → **serviço de ML em produção**

Ele mistura fundamentos de ciência de dados com práticas reais de engenharia, e por isso é muito valioso para portfólio, entrevistas e amadurecimento em MLOps.

O foco deve ser:

- fazer uma solução correta;
- mantê-la reproduzível;
- deixá-la testável;
- documentá-la com clareza;
- demonstrar maturidade operacional.

