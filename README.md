# Projeto MLlib — Previsão de Alta Rotatividade Agregada no Novo CAGED (2023–2025)

## 1. Objetivo do Projeto
Este projeto aplica técnicas de **Big Data, Processamento Distribuído e Aprendizagem de Máquina no PySpark (`pyspark.sql` e `pyspark.ml`)** sobre os microdados públicos do **Novo CAGED** no triênio **2023–2025**.

O objetivo central é identificar e classificar perfis socioeconômicos e setoriais (coortes agregadas) no mercado formal de trabalho segundo a propensão a apresentar **alta taxa de movimentações negativas (desligamentos) nos 6 meses subsequentes**.

---

## 2. Fonte dos Dados e Escopo Geográfico
- **Fonte:** Microdados públicos de movimentação mensal do Novo CAGED (Ministério do Trabalho e Emprego - MTE).
- **Período Observado:** 2023 a 2025 (35 competências mensais lidas).
- **Escopo de Análise:** **Região Nordeste** (AL, BA, CE, MA, PB, PE, PI, RN, SE), totalizando **18.996.006 registros de movimentação** processados.

---

## 3. Observação Metodológica Relevante
O objetivo original envolvia o acompanhamento individual de trabalhadores ao longo do tempo. No entanto, os microdados públicos do Novo CAGED omitem identificadores únicos de trabalhadores (como CPF e PIS por razões de privacidade), inviabilizando o rastreamento individual.

Para superar essa limitação sem comprometer o rigor científico, foi adota a **abordagem metodológica de Coortes Socioeconômicas Agregadas (`competênciamov + uf + seção + FAIXA_ETARIA`)**. Todas as análises e predições referem-se ao **comportamento futuro de perfis/coortes**, e **não** à previsão de demissão de trabalhadores individuais.

---

## 4. Tecnologias Utilizadas
- **Linguagem:** Python 3.10+
- **Processamento Distribuído:** Apache Spark / PySpark 3.3+ (`pyspark.sql`, `pyspark.ml`)
- **Gerenciador de Dependências:** `uv` (Fast Python package installer)
- **Interface:** Jupyter Notebook (`pipeline_mllib.ipynb`)
- **Armazenamento:** Apache Parquet (particionado por `ANO`)

---

## 5. Estrutura do Repositório
```text
caged-nordeste-mllib/
├── README.md                               # Documentação principal do projeto
├── notebooks/
│   └── pipeline_mllib.ipynb                # Notebook oficial com código, Markdown e outputs executados
├── data/
│   └── README.md                           # Orientações sobre fontes e política de dados brutos
├── silver/
│   └── caged_nordeste_ml/                  # Camada Silver tratada e particionada por ANO (Parquet)
├── models/
│   ├── logistic_regression_nordeste/       # Modelo salvo de Regressão Logística (PipelineModel)
│   ├── random_forest_nordeste/             # Modelo salvo de Random Forest Base (PipelineModel)
│   └── tuned_best_model_nordeste/          # Modelo salvo de Random Forest Tunada (PipelineModel)
├── outputs/
│   ├── metrics/
│   │   └── model_comparison_metrics.csv    # Relatório CSV consolidado das métricas de teste
│   ├── target_audit_nordeste/              # Intermediários de auditoria de target e leakage
│   └── rf_summary.json                     # Resumo das métricas e importâncias de atributos
├── scripts/
│   ├── build_silver_nordeste.py            # Construção e validação da camada Silver
│   ├── run_mllib_nordeste_pipeline.py      # Execução e avaliação da Logistic Regression
│   ├── run_rf_nordeste_pipeline.py         # Execução e avaliação da Random Forest
│   └── run_tvs_nordeste_tuning.py          # Tuning de hiperparâmetros via TrainValidationSplit
├── requirements.txt                        # Lista de dependências Python
├── pyproject.toml                          # Configuração do ambiente uv
├── uv.lock                                 # Lockfile de dependências congeladas
└── .gitignore                              # Regras de exclusão de arquivos pesados no Git
```

---

## 6. Como Executar o Projeto

### Pré-requisitos
Possuir Python 3.10+ e Java 8/11/17 instalado no sistema (necessário para a JVM do Apache Spark).

### Passo 1: Clonar o Repositório e Instalar Dependências
Utilizando `uv` (recomendado):
```bash
uv sync
```
Ou via `pip` tradicional:
```bash
pip install -r requirements.txt
```

### Passo 2: Executar o Notebook Jupyter
Inicie o servidor Jupyter Notebook:
```bash
uv run jupyter notebook notebooks/pipeline_mllib.ipynb
```
O notebook já contém **todos os outputs de código executados e salvos**.

---

## 7. Pipeline de Dados e Modelagem

```text
Arquivos Brutos (35 meses do Nordeste - 18,99M linhas)
   │
   ▼ (Filtragem Precoce na Leitura + Seleção de Colunas)
Camada Intermediária Mensal (outputs/nordeste_monthly/)
   │
   ▼ (Agregação por Coorte A: competênciamov + uf + seção + FAIXA_ETARIA)
Cálculo de Janelas Futuras de 6 Meses (202301 a 202506) & Censura à Direita
   │
   ▼ (Cálculo da Mediana P50_Nordeste = 0.479005 -> Target ALTA_ROTATIVIDADE_6M)
Camada Silver Particionada por ANO (silver/caged_nordeste_ml/)
   │
   ▼ (Split 70/30 com SEED 42 + StringIndexer + OneHotEncoder + VectorAssembler)
Matriz de Atributos (52 Dimensões)
   │
   ├─► Modelo 1: Logistic Regression (pyspark.ml.classification.LogisticRegression)
   ├─► Modelo 2: RandomForestClassifier (numTrees=30, maxDepth=8)
   └─► Tuning: TrainValidationSplit (ParamGridBuilder com 8 combinações)
```

---

## 8. Resultados Reais e Tabela Comparativa de Métricas

Avaliação dos modelos no conjunto de **Teste não visto (9.955 registros de coortes)**:

| Modelo | AUC-ROC | Acurácia | Precision (Classe 1) | Recall (Classe 1) | F1-Score | Especificidade | Tempo Treino |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (Classe Majoritária 0)** | N/A | 49.19% | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| **Logistic Regression** | 0.8726 | **78.90%** | **0.7704** | 0.8327 | 0.8004 | **0.7437** | 15.465s |
| **Random Forest Base (`numTrees=30`, `maxDepth=8`)** | **0.8813** | 78.56% | 0.7577 | **0.8499** | **0.8012** | 0.7192 | **9.605s** |
| **Random Forest Tunada (TVS - 8 Combinações)** | **0.8813** | 78.56% | 0.7577 | **0.8499** | **0.8012** | 0.7192 | 52.800s |

**Modelo Recomendado:** **`RandomForestClassifier`** por apresentar a maior capacidade discriminativa (**AUC-ROC de 0,8813**), o maior **Recall na classe de alta rotatividade (84,99%)** e o menor tempo de treinamento no PySpark.

---

## 9. Limitações Conhecidas
1. **Ausência de CPF/ID Único:** Rastreamento limitado à dinâmica agregada de coortes socioeconômicas.
2. **Exclusividade do Mercado Formal:** O CAGED registra exclusivamente contratos CLT, omitindo a economia informal.
3. **Censura à Direita:** Limitação da janela de referência até 202506 para garantir a observação completa dos 6 meses futuros até 202512.
4. **Divisão Aleatória (`randomSplit`):** Pode apresentar estimativas otimistas quando comparada a uma validação por split temporal estrito.
