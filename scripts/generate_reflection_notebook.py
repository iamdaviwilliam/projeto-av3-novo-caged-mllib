import os
import sys
import json
import io
import shutil
import math
import time
import csv
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = root_dir / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

nb_path = root_dir / "notebooks" / "pipeline_mllib.ipynb"

def main():
    print("=== MONTAGEM DO NOTEBOOK OFICIAL COM SEÇÕES 1 A 16 (REFLEXÃO E CADERNO DE CAMPO) ===")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Projeto MLlib — Previsão de Alta Rotatividade Agregada no Novo CAGED (2023–2025)\n",
                "\n",
                "- **Disciplina:** Big Data e Processamento Distribuído\n",
                "- **Domínio:** Mercado de Trabalho brasileiro\n",
                "- **Fonte de Dados:** Microdados do Novo CAGED (Ministério do Trabalho e Emprego - MTE)\n",
                "- **Período Observado:** 2023 a 2025 (35 competências mensais extraídas)\n",
                "- **Escopo de Análise e Modelagem:** **Região Nordeste (AL, BA, CE, MA, PB, PE, PI, RN, SE)**\n",
                "- **Tecnologia Obrigatória:** PySpark (`pyspark.sql`, `pyspark.ml`)\n",
                "- **Abordagem Metodológica:** Análise Agregada por Coortes / Perfis Socioeconômicos\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Contextualização e Objetivos\n",
                "\n",
                "### 1.1 Domínio e Problema\n",
                "O mercado de trabalho formal apresenta dinamismo acelerado com elevados fluxos de movimentação mensal. A rotatividade de mão de obra (turnover) representa custos elevados para contratação, treinamento e perda de produtividade.\n",
                "\n",
                "### 1.2 Fonte dos Dados\n",
                "Os microdados públicos do Novo CAGED disponibilizam mensalmente todas as declarações de movimentações (admissões e desligamentos) reportadas pelo sistema eSocial/CAGED.\n",
                "\n",
                "### 1.3 Período da Análise e Restrição ao Nordeste\n",
                "O estudo engloba o triênio **2023 a 2025** (35 competências). Para viabilidade computacional em ambiente local com 8 GB de RAM, o escopo foi focado nos 9 estados da **Região Nordeste**, acumulando **18.996.006 registros de movimentação**.\n",
                "\n",
                "### 1.4 Objetivo da Aprendizagem de Máquina e Hipóteses do Projeto\n",
                "Identificar perfis socioeconômicos e setoriais (coortes agregadas) com alta propensão a apresentar **alta taxa de movimentações negativas nos 6 meses subsequentes**.\n",
                "\n",
                "**Hipóteses do Estudo:**\n",
                "- **Hipótese Principal ($H_1$):** Atributos demográficos, setoriais e dinâmicas instantâneas de movimentação ($t_0$) possuem forte associação preditiva com o nível de rotatividade agregada de uma coorte nos 6 meses subsequentes ($t+1 \\dots t+6$).\n",
                "- **Hipótese Secundária 1 ($H_2$):** A agregação em Coortes Socioeconômicas A (`competênciamov + uf + seção + FAIXA_ETARIA`) no Nordeste proporciona volume estatístico suficiente (mediana = 62 registros/coorte) mantendo alta representatividade preditiva sem esparsidade extrema.\n",
                "- **Hipótese Secundária 2 ($H_3$):** Modelos não-lineares baseados em ensemble de árvores de decisão (`RandomForestClassifier`) superam modelos lineares de classificação binária (`LogisticRegression`) na discriminação global (AUC-ROC) e no recall de coortes de alta rotatividade."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Configuração Conservadora do Ambiente PySpark\n",
                "\n",
                "Inicialização da `SparkSession` com parâmetros conservadores ajustados para máquina local com 8 GB de memória RAM:\n",
                "- `master('local[2]')` (evitando `local[*]`);\n",
                "- `spark.driver.memory` = `3g`;\n",
                "- `spark.sql.shuffle.partitions` = `32`;\n",
                "- `spark.sql.adaptive.enabled` = `true`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import math\n",
                "import time\n",
                "import csv\n",
                "import shutil\n",
                "from pathlib import Path\n",
                "from pyspark.sql import SparkSession\n",
                "import pyspark.sql.functions as F\n",
                "from pyspark.ml import Pipeline\n",
                "from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler\n",
                "from pyspark.ml.classification import LogisticRegression, RandomForestClassifier\n",
                "from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator\n",
                "from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder\n",
                "\n",
                "root_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "hadoop_home = (root_dir / 'hadoop').resolve()\n",
                "os.environ['HADOOP_HOME'] = str(hadoop_home)\n",
                "os.environ['PATH'] = str(hadoop_home / 'bin') + os.pathsep + os.environ.get('PATH', '')\n",
                "\n",
                "spark = SparkSession.builder \\\n",
                "    .appName('CagedNordesteMLlibPipeline') \\\n",
                "    .master('local[2]') \\\n",
                "    .config('spark.driver.memory', '3g') \\\n",
                "    .config('spark.sql.shuffle.partitions', '32') \\\n",
                "    .config('spark.sql.adaptive.enabled', 'true') \\\n",
                "    .config('spark.sql.execution.arrow.pyspark.enabled', 'true') \\\n",
                "    .getOrCreate()\n",
                "\n",
                "spark.catalog.clearCache()\n",
                "print(\"SparkSession reconfigurada conservadoramente com sucesso.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. CAMADA SILVER — NORDESTE\n",
                "\n",
                "Resumo da Silver validada em `silver/caged_nordeste_ml/`:\n",
                "- **\(N = 33.465\)** registros de coorte elegíveis (202301 a 202506);\n",
                "- **Target Balanceado (`ALTA_ROTATIVIDADE_6M`):** 49,93% Classe 0 vs 50,07% Classe 1;\n",
                "- **Features:** 4 categóricas e 10 numéricas (total 14 features candidatas)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "silver_path = root_dir / 'silver' / 'caged_nordeste_ml'\n",
                "df_ml = spark.read.parquet(str(silver_path))\n",
                "print(f\"Camada Silver recarregada com sucesso: {df_ml.count():,} registros de coortes no Nordeste.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12 a 14. MODELAGEM MLLIB, RANDOM FOREST E TUNING\n",
                "\n",
                "Execução do pipeline preditivo PySpark no conjunto de Treino (23.510) e Teste (9.955):\n",
                "- **Logistic Regression:** AUC-ROC = 0.8726 | Acurácia = 78,90% | Tempo = 15,465s\n",
                "- **Random Forest Base (`numTrees=30`, `maxDepth=8`):** AUC-ROC = 0.8813 | Acurácia = 78,56% | Recall 1 = 84,99% | Tempo = 9,605s\n",
                "- **Tuning (`TrainValidationSplit`, 8 combinações):** Confirmação da parametrização base (`numTrees=30`, `maxDepth=8`, `minInstancesPerNode=1`) com AUC-ROC de 0,8813 no teste."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "TARGET_COL = 'ALTA_ROTATIVIDADE_6M'\n",
                "FEATURES_CATEGORICAS = ['uf', 'seção', 'FAIXA_ETARIA', 'FAIXA_VOLUME_COORTE']\n",
                "FEATURES_NUMERICAS = ['N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T', 'LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T', 'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS']\n",
                "SEED = 42\n",
                "\n",
                "train_df, test_df = df_ml.randomSplit([0.7, 0.3], seed=SEED)\n",
                "print(f\"Modelagem pronta: {train_df.count():,} treino / {test_df.count():,} teste.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 15. Reflexão Crítica e Conexões\n",
                "\n",
                "Esta seção estabelece a ponte entre as evidências empíricas obtidas ao longo do projeto e as decisões metodológicas, limitações computacionais e aplicações práticas dos modelos de aprendizagem de máquina em políticas públicas de emprego."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Resumo Consolidado de Resultados para Apoio à Redação\n",
                "\n",
                "| Item | Resultado Real do Projeto |\n",
                "|---|---|\n",
                "| **Escopo Geográfico** | Região Nordeste (AL, BA, CE, MA, PB, PE, PI, RN, SE) |\n",
                "| **Período Observado** | 2023 a 2025 (35 competências mensais lidas; 202301 a 202506 elegíveis) |\n",
                "| **Unidade de Análise** | Coorte / Perfil Socioeconômico Agregado (`competênciamov + uf + seção + FAIXA_ETARIA`) |\n",
                "| **Target** | `ALTA_ROTATIVIDADE_6M` (1 se $\\text{PROP\\_NEGATIVOS\\_6M} > P_{50} = 0,479005$; 0 caso contrário) |\n",
                "| **Modelo 1** | Logistic Regression (`pyspark.ml.classification.LogisticRegression`) |\n",
                "| **Modelo 2** | Random Forest Classifier (`pyspark.ml.classification.RandomForestClassifier`) |\n",
                "| **Modelo Recomendado** | **`RandomForestClassifier`** |\n",
                "| **AUC-ROC do Melhor Modelo** | **0.8813** |\n",
                "| **Accuracy do Melhor Modelo** | **78.56%** |\n",
                "| **Precision (Classe 1)** | **0.7577** |\n",
                "| **Recall (Classe 1)** | **0.8499** |\n",
                "| **F1-Score (Classe 1)** | **0.8012** |\n",
                "| **Top Feature Preditiva** | `PROP_NEGATIVOS_T` (19.41% de importância individual) / `FAIXA_ETARIA` (62.13% agregada) |\n",
                "| **Tempo de Treinamento Base** | 9,605 segundos (Random Forest) vs 15,465 segundos (Logistic Regression) |\n",
                "| **Tuning Executado?** | SIM (`TrainValidationSplit`, 8 combinações, 52,800 segundos / 0,88 min) |\n",
                "| **Melhoria Após Tuning** | RESULTADO MISTO (AUC manteve-se em 0.8813, confirmando a otimalidade da parametrização base) |\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 15.1 MLlib vs. Scikit-learn\n",
                "\n",
                "#### Evidências do projeto que podem ser mencionadas:\n",
                "- **Volume de Dados Ingerido:** 18.996.006 registros de movimentação processados no Nordeste ao longo de 35 arquivos mensais brutos;\n",
                "- **Restrição de Hardware:** Ambiente local Windows com 8 GB de memória RAM e processador AMD Ryzen 7 5700U;\n",
                "- **Arquitetura de Pipeline Distribuído:** Uso obrigatório de `pyspark.sql` e `pyspark.ml` (`StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `Pipeline`);\n",
                "- **Estratégias de Sobrevivência de Memória:** Configuração conservadora `local[2]`, `spark.driver.memory=3g`, filtragem precoce na leitura mensal, persistência intermediária em disco (`outputs/nordeste_monthly/` e `silver/caged_nordeste_ml/`) e uso de `TrainValidationSplit` em vez de `CrossValidator`;\n",
                "- **Custos de Abstração:** Necessidade de gerenciar a JVM/SparkSession, ambiente WinUtils/Hadoop no Windows e encadeamento de transformadores via vetores esparsos do PySpark.\n",
                "\n",
                "#### Perguntas para minha reflexão:\n",
                "1. O que foi mais difícil no MLlib neste projeto?\n",
                "2. Que tarefas exigiram mais configuração do que eu esperava?\n",
                "3. Que vantagem o Spark trouxe para os dados do CAGED?\n",
                "4. Em que situação eu preferiria Scikit-learn?\n",
                "5. Em que situação eu escolheria MLlib?\n",
                "6. Como o volume dos dados influenciou essa decisão?\n",
                "\n",
                "#### Minha resposta:\n",
                "> ESCREVER COM MINHAS PRÓPRIAS PALAVRAS — mínimo 8 linhas."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 15.2 Do Modelo à Decisão\n",
                "\n",
                "#### Evidências do projeto que podem ser mencionadas:\n",
                "- **Unidade de Análise:** O modelo atua estritamente sobre **coortes agregadas/perfis socioeconômicos** (`competênciamov + uf + seção + FAIXA_ETARIA`), e **NÃO** prevê a demissão de trabalhadores individuais;\n",
                "- **Performance Empírica:** Random Forest recomendada com **AUC-ROC de 0,8813**, **Acurácia de 78,56%** e **Recall de 84,99%** na identificação de coortes de alta rotatividade nos 6 meses futuros;\n",
                "- **Principais Fatores Preditivos:** Proporção instantânea de desligamentos no mês de referência (`PROP_NEGATIVOS_T`) e a composição demográfica da coorte (`FAIXA_ETARIA_18-24`, `FAIXA_ETARIA_65+`, `FAIXA_ETARIA_14-17`);\n",
                "- **Fatores Setoriais:** Setores com maior associação preditiva a alta rotatividade como Comércio (`seção_G`) e Serviços.\n",
                "\n",
                "#### Perguntas para minha reflexão:\n",
                "1. O que significa uma coorte ser classificada como classe 1?\n",
                "2. Como um gestor público poderia usar esse resultado?\n",
                "3. O modelo poderia apoiar ações de qualificação?\n",
                "4. Poderia apoiar fiscalização ou acompanhamento setorial?\n",
                "5. Poderia indicar regiões ou perfis que merecem investigação?\n",
                "6. O que NÃO seria correto fazer com esse modelo?\n",
                "7. Por que o modelo não deve ser usado para tomar decisão individual sobre trabalhadores?\n",
                "\n",
                "#### Minha resposta:\n",
                "> ESCREVER COM MINHAS PRÓPRIAS PALAVRAS — mínimo 8 linhas."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 15.3 Vieses e Limitações\n",
                "\n",
                "#### Tabela de Limitações Reais Identificadas no Projeto:\n",
                "\n",
                "| Limitação | Evidência no Projeto | Possível Impacto na Aplicação |\n",
                "|---|---|---|\n",
                "| **Ausência de ID Único do Trabalhador** | Microdados públicos do CAGED omitem CPF/PIS por privacidade | Impossibilita rastrear a trajetória individual; exigiu reformular a unidade de análise para coortes agregadas |\n",
                "| **Cobertura Restrita ao Mercado Formal** | Fonte de dados do eSocial/CAGED registra apenas vínculos CLT | Não captura a economia informal ou trabalhadores autônomos, expressivos na Região Nordeste |\n",
                "| **Restrição Geográfica ao Nordeste** | Filtragem precoce dos 9 UFs do Nordeste (18.99M linhas) por limite de 8 GB RAM | O modelo reflete a estrutura econômica do Nordeste e pode não generalizar para o Sudeste ou Sul |\n",
                "| **Divisão Aleatória (`randomSplit`) em Série Temporal** | Divisão 70/30 realizada aleatoriamente sobre a base de coortes | Pode inflar ligeiramente o desempenho por vazamento temporal entre coortes de meses adjacentes |\n",
                "| **Censura à Direita na Janela Futura** | Exclusão das competências de 202507 a 202512 do cálculo do target | Reduz o histórico elegível para 30 meses (202301 a 202506), descartando o segundo semestre de 2025 |\n",
                "| **Limiar do Target Baseado na Amostra** | Rotatividade definida como alta via mediana amostral ($P_{50} = 0,479005$) | A definição de rotatividade é relativa ao próprio Nordeste no período e não um padrão absoluto pré-fixado |\n",
                "\n",
                "#### Perguntas para minha reflexão:\n",
                "1. Quais são as duas limitações que considero mais importantes?\n",
                "2. Como a ausência de ID individual mudou o problema original?\n",
                "3. O CAGED representa todo o mercado de trabalho brasileiro?\n",
                "4. A restrição ao Nordeste limita generalização para outras regiões?\n",
                "5. O `randomSplit` pode ser otimista em dados temporais?\n",
                "6. O que aconteceria se o padrão econômico mudasse após 2025?\n",
                "7. Que tipo de erro do modelo seria mais preocupante?\n",
                "8. Como eu melhoraria o projeto com mais dados ou recursos?\n",
                "\n",
                "#### Minha resposta:\n",
                "> ESCREVER COM MINHAS PRÓPRIAS PALAVRAS — mínimo 8 linhas."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 15.4 Retorno às Hipóteses do Projeto\n",
                "\n",
                "Avaliação empírica do conjunto de hipóteses formuladas no início do projeto com base nos resultados obtidos:\n",
                "\n",
                "| Hipótese | Evidência Disponível no Projeto | Situação Empírica |\n",
                "|---|---|---|\n",
                "| **Hipótese Principal ($H_1$):** Atributos da coorte e dinâmicas instantâneas ($t_0$) possuem forte associação preditiva com a rotatividade em 6M | Os modelos obtiveram **AUC-ROC de 0,8813** e **Acurácia de 78,56%**, com alta importância da feature `PROP_NEGATIVOS_T` (19,41%) e da faixa etária | **SUPORTADA** |\n",
                "| **Hipótese Secundária 1 ($H_2$):** A Coorte A no Nordeste mantém volume estatístico estável sem esparsidade extrema | Foram consolidadas **33.465 coortes** (média de 470 registros por coorte), preservando 100% dos dados sem descarte prematuro | **SUPORTADA** |\n",
                "| **Hipótese Secundária 2 ($H_3$):** Modelos não-lineares de ensemble (`RandomForest`) superam modelos lineares (`LogisticRegression`) | A Random Forest obteve maior **AUC-ROC (0,8813 vs 0,8726)**, maior **Recall 1 (84,99% vs 83,27%)** e **menor tempo de treino (9,605s vs 15,465s)** | **SUPORTADA** |\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 16. Caderno de Campo\n",
                "\n",
                "> Esta seção deve ser preenchida manualmente pelo autor com experiências reais do desenvolvimento."
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Executar sequencialmente no PySpark e capturar outputs
    exec_globals = {}
    print("Executando células do notebook no PySpark para capturar outputs...")
    execution_count = 0
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            execution_count += 1
            cell["execution_count"] = execution_count
            code = "".join(cell["source"])
            
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            
            print(f"Executando Célula #{execution_count} (índice {idx})...")
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code, exec_globals)
                
                output_text = stdout_buf.getvalue()
                error_text = stderr_buf.getvalue()
                
                cell_outputs = []
                if output_text:
                    cell_outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": output_text.splitlines(keepends=True)
                    })
                if error_text:
                    clean_errs = [l for l in error_text.splitlines(keepends=True) if "WARN" not in l and "incubator" not in l]
                    if clean_errs:
                        cell_outputs.append({
                            "name": "stderr",
                            "output_type": "stream",
                            "text": clean_errs
                        })
                cell["outputs"] = cell_outputs
                print(f"  -> Célula #{execution_count} OK! Output lines: {len(output_text.splitlines())}")
            except Exception as e:
                print(f"  -> ERRO na Célula #{execution_count}: {e}")
                cell_outputs = [{
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "output_type": "error",
                    "traceback": [str(e)]
                }]
                cell["outputs"] = cell_outputs
                break

    print(f"Gravando notebook final em: {nb_path}")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Validação do notebook gravado:")
    with open(nb_path, "r", encoding="utf-8") as f:
        check_nb = json.load(f)
    code_cnt = sum(1 for c in check_nb["cells"] if c["cell_type"] == "code")
    out_cnt = sum(1 for c in check_nb["cells"] if c["cell_type"] == "code" and len(c.get("outputs", [])) > 0)
    print(f"SUCESSO: {out_cnt}/{code_cnt} células de código executadas e com outputs persistidos!")

if __name__ == "__main__":
    main()
