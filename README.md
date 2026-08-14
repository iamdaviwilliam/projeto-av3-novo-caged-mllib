# Projeto AV3 — Previsão de Alta Rotatividade Agregada no Novo CAGED com PySpark MLlib

Análise distribuída dos dados do Novo CAGED de 2023 a 2025 com foco na Região Nordeste.

---

## 1. Contexto Acadêmico

- **Disciplina:** Big Data e Processamento Distribuído
- **Avaliação:** Terceira Avaliação (AV3)
- **Domínio:** Mercado de Trabalho Brasileiro (Ministério do Trabalho e Emprego — MTE)
- **Tema:** Previsão de Alta Rotatividade Agregada de Mão de Obra no Novo CAGED (2023–2025)
- **Abordagem Computacional:** Processamento Distribuído e Aprendizagem de Máquina no PySpark (`pyspark.sql` e `pyspark.ml`)
- **Escopo Geográfico:** **Região Nordeste** (AL, BA, CE, MA, PB, PE, PI, RN, SE)

---

## 2. Objetivo do Projeto

O objetivo principal deste projeto é utilizar o ecossistema Apache Spark e PySpark MLlib para identificar e classificar perfis socioeconômicos e setoriais (coortes agregadas) com alta propensão a apresentar **alta taxa de movimentações negativas (desligamentos) nos 6 meses subsequentes**.

O projeto visa fornecer um instrumento analítico preditivo para apoio ao planejamento de políticas públicas de trabalho, qualificação profissional e intermediação de mão de obra na Região Nordeste.

---

## 3. Adaptação Metodológica

A proposta original do estudo previa a identificação individual da probabilidade de um trabalhador admitido ser desligado em menos de seis meses. Contudo, a auditoria dos microdados públicos do Novo CAGED revelou que **as tabelas não contêm um identificador único e persistente** (como CPF mascarado ou PIS) por razões de sigilo e privacidade.

Para contornar essa restrição sem comprometer o rigor científico:
1. **Rejeição de Chaves Pseudônimas Artificiais:** Avaliou-se a criação de chaves compostas (ex.: `idade + sexo + município + CBO + salário`). Essa abordagem foi **estritamente rejeitada** devido ao risco de colisões entre pessoas distintas, o que geraria falsas associações longitudinais.
2. **Abordagem Agregada por Coortes Socioeconômicas:** O problema foi formalmente redefinido: a unidade fundamental de análise passou a ser a **Coorte Socioeconômica e Setorial** (`competênciamov + uf + seção + FAIXA_ETARIA`).

> **ESCLARECIMENTO RIGOROSO:**
> O modelo desenvolvido neste trabalho **NÃO prevê o desligamento de trabalhadores individuais** nem estima o risco de demissão de uma pessoa física. O modelo prevê a **intensidade relativa de movimentações negativas de uma Coorte Agregada** nos 6 meses futuros.

---

## 4. Dados Utilizados

- **Fonte:** Microdados públicos mensais do Novo CAGED (Ministério do Trabalho e Emprego — MTE).
- **Arquivos Auditados:** `CAGEDMOV` (movimentações), `CAGEDEXC` (exclusões) e `CAGEDFOR` (fora do prazo).
- **Período Lido:** Triênio **2023 a 2025** (**35 competências mensais** extraídas de `202301` a `202512`, exceto a competência `202312` não disponibilizada na fonte original).
- **Escopo Geográfico:** 9 estados da Região Nordeste (Alagoas, Bahia, Ceará, Maranhão, Paraíba, Pernambuco, Piauí, Rio Grande do Norte e Sergipe).
- **Volumetria Processada:** **18.996.006 registros de movimentação** na Região Nordeste.
- **Período Elegível do Target:** Competências `202301` a `202506` (**33.465 coortes elegíveis**).
- **Censura à Direita:** Como o indicador futuro analisa 6 meses subsequentes ($t+1 \dots t+6$), a última competência de referência elegível é **`202506` (Junho de 2025)**, pois exige observação até `202512`. Coortes a partir de `202507` foram desconsideradas do cálculo do target por falta de janela futura completa.

---

## 5. Estratégia de Processamento Local (8 GB RAM)

O processamento nacional do Novo CAGED engloba mais de 118 milhões de movimentações. Para viabilizar a execução em um ambiente computacional local restrito a **8 GB de memória RAM**, adotaram-se as seguintes otimizações no PySpark:
1. **Filtragem Precoce (*Pushdown Filter*):** Filtragem estrita dos códigos de UF do Nordeste logo na leitura inicial dos arquivos CSV/TXT;
2. **Projeção Seletiva de Colunas:** Manutenção apenas dos atributos essenciais para a formação das coortes;
3. **Agregação em Dois Níveis:** Redução da cardinalidade antes da aplicação de junções temporais;
4. **Persistência Intermediária em Parquet:** Gravação de dados tratados em `outputs/nordeste_monthly/` e `silver/caged_nordeste_ml/` particionados por `ANO`, evitando recálculos da CPU e estresse de RAM;
5. **Configuração Otimizada da SparkSession:**
   - `.master('local[2]')` (limita a 2 trabalhadores locais);
   - `spark.driver.memory = '3g'` (alocação controlada para o driver);
   - `spark.sql.shuffle.partitions = '32'` (reduz overhead de particionamento);
   - `spark.sql.adaptive.enabled = 'true'` (ativa execução adaptativa de queries).

---

## 6. Unidade de Análise — Coortes Socioeconômicas

A unidade de análise é a **Coorte Socioeconômica**, definida pela chave primária exata:

$$\text{Chave da Coorte} = \text{competênciamov} + \text{uf} + \text{seção} + \text{FAIXA\_ETARIA}$$

- **`competênciamov`:** Mês de referência contemporâneo ($t_0$);
- **`uf`:** Estado da Região Nordeste;
- **`seção`:** Seção Econômica da CNAE 2.0 (ex.: Indústria, Comércio, Serviços);
- **`FAIXA_ETARIA`:** Faixa etária do grupo (`<18`, `18-24`, `25-34`, `35-49`, `50-64`, `65+`).

> **EXEMPLO DIDÁTICO ESTRUTURAL:**
> `202401` (Janeiro/2024) + `PB` (Paraíba) + `Seção G` (Comércio) + `25-34` (Faixa Etária)
> $\rightarrow$ Representa 1 única linha agregada no conjunto de dados de modelagem.

---

## 7. Indicador Principal — `PROP_NEGATIVOS_6M`

O principal indicador construído no projeto é o **`PROP_NEGATIVOS_6M`**, que mede a intensidade acumulada de desligamentos da coorte no horizonte futuro de 6 meses:

$$\text{PROP\_NEGATIVOS\_6M} = \frac{\text{N\_NEGATIVOS\_6M}}{\text{N\_TOTAL\_6M}}$$

- **Numerador ($\text{N\_NEGATIVOS\_6M}$):** Soma de todos os desligamentos/movimentos negativos ($saldomovimentação = -1$) observados na janela futura de 6 meses ($t+1 \dots t+6$) para a mesma coorte (`uf + seção + FAIXA_ETARIA`).
- **Denominador ($\text{N\_TOTAL\_6M}$):** Soma de todas as movimentações totais ($saldomovimentação = +1 \text{ ou } -1$) observadas na janela futura de 6 meses ($t+1 \dots t+6$) para a mesma coorte.

### Estatísticas Descritivas e Mediana $P_{50}$
A distribuição empírica do indicador contínuo na Região Nordeste apresentou a seguinte mediana histórica:

$$\text{Mediana } P_{50} \approx 0,479005$$

> **INTERPRETAÇÃO RIGOROSA DO $P_{50}$:**
> O valor **0,479005** corresponde à mediana da distribuição do indicador contínuo entre as 33.465 coortes elegíveis.
> **NÃO** significa que "47,9% dos trabalhadores foram demitidos", e **NÃO** representa uma taxa individual de desligamento.

---

## 8. Construção do Target (`ALTA_ROTATIVIDADE_6M`)

O indicador contínuo foi binarizado com base na mediana $P_{50}$ para alimentar os algoritmos de classificação binária do PySpark MLlib:

$$\text{ALTA\_ROTATIVIDADE\_6M} = \begin{cases} 1 & \text{se } \text{PROP\_NEGATIVOS\_6M} > P_{50} \text{ (0,479005)} \\ 0 & \text{se } \text{PROP\_NEGATIVOS\_6M} \le P_{50} \text{ (0,479005)} \end{cases}$$

- **Classe 1:** Coortes com proporção de desligamentos futuros **acima da mediana** (maior intensidade relativa de saídas);
- **Classe 0:** Coortes com proporção de desligamentos futuros **menor ou igual à mediana** (menor rotatividade agregada relativa);
- **Distribuição de Classes:** **49,93% Classe 0** vs **50,07% Classe 1** (Target perfeitamente equilibrado).

---

## 9. Features e Auditoria de Data Leakage

### Matriz de Features (52 Dimensões)
O vetor de entrada `"features"` foi construído via `VectorAssembler` a partir de 14 variáveis originais:
- **Features Categóricas (4):** `uf`, `seção`, `FAIXA_ETARIA`, `FAIXA_VOLUME_COORTE` (codificadas via `StringIndexer` + `OneHotEncoder`).
- **Features Numéricas e Temporais (10):** `N_TOTAL_T`, `N_POSITIVOS_T`, `N_NEGATIVOS_T`, `LOG_VOLUME_COORTE` ($\log(1 + N\_TOTAL\_T)$), `PROP_NEGATIVOS_T` ($N\_NEGATIVOS\_T / N\_TOTAL\_T$), `ANO`, `MES`, `TRIMESTRE`, `MES_SIN` ($\sin(2\pi \cdot MES / 12)$), `MES_COS` ($\cos(2\pi \cdot MES / 12)$).

### Auditoria de Temporalidade e Data Leakage
Todas as variáveis pertencentes à janela futura ($t+1 \dots t+6$), como `N_NEGATIVOS_6M`, `N_POSITIVOS_6M`, `N_TOTAL_6M` e o próprio indicador `PROP_NEGATIVOS_6M`, foram **estritamente isoladas e proibidas de entrar no modelo como preditores**, sendo utilizadas exclusivamente na construção da variável dependente (`ALTA_ROTATIVIDADE_6M`).

---

## 10. Resultados e Tabela Comparativa de Métricas

Os modelos foram treinados em 70% das coortes (`train_df` = 23.510 registros) e avaliados sobre o conjunto de **Teste isolado (30% - 9.955 registros de coortes)**:

| Modelo | AUC-ROC | Acurácia | Precision (Classe 1) | Recall (Classe 1) | F1-Score | Especificidade | Tempo Treino |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (Classe Majoritária 0)** | N/A | 49.19% | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| **Logistic Regression** | 0.8726 | **78.90%** | **0.7704** | 0.8327 | 0.8004 | **0.7437** | ~16.0s |
| **Random Forest Base (`numTrees=30`, `maxDepth=8`)** | **0.8813** | 78.56% | 0.7577 | **0.8499** | **0.8012** | 0.7192 | **~9.3s** |
| **Random Forest Tunada (TVS — 8 Combinações)** | **0.8813** | 78.56% | 0.7577 | **0.8499** | **0.8012** | 0.7192 | ~52.8s |

> **MODELO RECOMENDADO:** **`RandomForestClassifier`**
> O modelo de floresta aleatória apresentou a maior capacidade de discriminação global (**AUC-ROC = 0,8813**), o maior **Recall na classe de alta rotatividade (84,99%)** e o menor tempo de treinamento no PySpark.

---

## 11. Interpretação de Feature Importance e Análise CBO

### Importância de Atributos na Random Forest
- **`PROP_NEGATIVOS_T`:** Responde isoladamente por **19,41%** da importância do modelo, mostrando que o comportamento contemporâneo no mês $t$ possui forte poder associativo com a rotatividade futura.
- **`FAIXA_ETARIA`:** Responde de forma agregada por **62,13%** da importância preditiva, evidenciando forte variação do comportamento de turnover conforme o segmento demográfico.

### Análise Complementar de Ocupações CBO
A análise dos grupos ocupacionais dominantes (`SHARE_CBO_DOMINANTE`) via PySpark Window Functions confirmou que a inclusão das dimensões de Seção Econômica, UF e Faixa Etária na Coorte A atua como uma proxy agregada eficaz para capturar as oscilações setoriais sem necessitar de codificar 2.562 códigos OHE adicionais.

---

## 12. Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Processamento Distribuído:** Apache Spark / PySpark 3.3+ (`pyspark.sql`, `pyspark.ml`)
- **Gerenciador de Ambientes e Dependências:** `uv` (Fast Python package installer)
- **Interface e Relatório:** Jupyter Notebook (`pipeline_mllib.ipynb`)
- **Formatos de Armazenamento:** Apache Parquet (particionado por `ANO`)
- **Visualização de Dados:** Matplotlib & Seaborn

---

## 13. Estrutura do Repositório

```text
av3bigdata/
├── README.md                               # Documentação principal do projeto
├── README_backup_pre_final.md             # Backup da documentação
├── notebooks/
│   ├── pipeline_mllib.ipynb                # Notebook oficial consolidado com código, comentários e outputs
│   └── pipeline_mllib_backup_antes_comentarios.ipynb
├── silver/
│   └── caged_nordeste_ml/                  # Camada Silver tratada e particionada por ANO (Parquet)
├── outputs/
│   ├── metrics/
│   │   └── model_comparison_metrics.csv    # Relatório CSV de métricas
│   ├── nordeste_monthly/                   # Checkpoints Parquet mensais agregados
│   ├── target_audit_nordeste/              # Parquets de auditoria de target e modelagem
│   └── mllib_summary.json                  # Resumo JSON das métricas executadas
├── scripts/
│   ├── enrich_pipeline_notebook.py         # Script de estruturação do notebook
│   ├── add_code_comments.py                # Script de adição dos comentários didáticos
│   ├── build_silver_nordeste.py            # Construção e validação da camada Silver
│   ├── run_mllib_nordeste_pipeline.py      # Pipeline da Regressão Logística
│   └── run_rf_nordeste_pipeline.py         # Pipeline da Random Forest
├── pyproject.toml                          # Configuração de dependências via uv
├── requirements.txt                        # Arquivo de dependências padrão
└── uv.lock                                 # Lockfile de dependências
```

---

## 14. Como Executar o Projeto

### Pré-requisitos
- Python 3.10 ou superior
- Java OpenJDK 8, 11 ou 17 (necessário para a execução do Apache Spark / JVM)

### Passo 1: Instalar Dependências
Utilizando o `uv` (recomendado):
```bash
uv sync
```
Ou via `pip` tradicional:
```bash
pip install -r requirements.txt
```

### Passo 2: Iniciar o Jupyter Notebook
```bash
uv run jupyter notebook notebooks/pipeline_mllib.ipynb
```
O notebook `notebooks/pipeline_mllib.ipynb` contém **todos os outputs de código executados, visualizações e relatórios gravados**.

---

## 15. Limitações Conhecidas e Recomendações

1. **Ausência de CPF/ID Único:** Restringe a análise à dinâmica agregada de coortes socioeconômicas.
2. **Foco no Mercado Formal:** Os dados do Novo CAGED englobam exclusivamente contratos com carteira assinada (CLT), não cobrindo o mercado informal.
3. **Censura à Direita:** Limita o período de referência até `202506` para garantir a observação completa dos 6 meses subsequentes até `202512`.
4. **Utilização do Modelo:** Recomenda-se o uso do modelo exclusivamente para apoio à gestão pública e políticas coletivas de trabalho. É **vedada** sua utilização para avaliações, contratações, demissões ou punições individuais.
