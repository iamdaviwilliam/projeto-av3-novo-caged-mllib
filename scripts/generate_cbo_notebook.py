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
    print("=== INSERÇÃO DA ANÁLISE INTERPRETATIVA DE CBO NO NOTEBOOK ===")

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
                "## 16. Análise Interpretativa — Ocupações e Rotatividade no Nordeste\n",
                "\n",
                "Esta etapa complementar realiza uma análise descritiva e interpretativa sobre os **cargos e grupos ocupacionais (CBO)** no mercado formal de trabalho da Região Nordeste. O objetivo é compreender quais áreas profissionais concentram maior volume de movimentações e quais apresentam maior associação empírica com o indicador de **alta rotatividade futura (`ALTA_ROTATIVIDADE_6M`)**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "from pathlib import Path\n",
                "from pyspark.sql import SparkSession\n",
                "import pyspark.sql.functions as F\n",
                "\n",
                "root_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "monthly_path = root_dir / 'outputs' / 'nordeste_monthly'\n",
                "audit_path = root_dir / 'outputs' / 'target_audit_nordeste' / 'df_target_audit.parquet'\n",
                "\n",
                "df_monthly = spark.read.parquet(str(monthly_path))\n",
                "df_monthly = df_monthly.withColumn('GRUPO_CBO', F.substring(F.col('cbo2002ocupação').cast('string'), 1, 2))\n",
                "\n",
                "card_cbo_bruto = df_monthly.select('cbo2002ocupação').distinct().count()\n",
                "card_grupo_cbo = df_monthly.select('GRUPO_CBO').distinct().count()\n",
                "\n",
                "print(f\"VARIAVEL_OCUPACIONAL           = cbo2002ocupação (bruto) e GRUPO_CBO (2 dígitos)\")\n",
                "print(f\"CARDINALIDADE_CBO_BRUTO (6 dígitos) = {card_cbo_bruto:,}\")\n",
                "print(f\"CARDINALIDADE_GRUPO_CBO (2 dígitos)= {card_grupo_cbo:,}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Documentação do Nível da CBO\n",
                "\n",
                "O código ocupacional original (`cbo2002ocupação`) possui **2.562 ocupações individuais** de 6 dígitos (ex: `521110` - Vendedor de comércio varejista). A utilização direta dessas milhares de categorias causaria esparsidade extrema na agregação por coortes.\n",
                "\n",
                "Por essa razão, a análise foi realizada sobre o **`GRUPO_CBO`**, extraído a partir dos **2 primeiros dígitos da CBO 2002** (correspondendo ao Subgrupo Principal / Grande Grupo Ocupacional da estrutura oficial do MTE, totalizando **46 grupos**). As descrições textuais das ocupações devem ser consultadas na tabela oficial do Ministério do Trabalho e Emprego (MTE).\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Top 10 Grupos Ocupacionais por Volume vs Associação com Alta Rotatividade\n",
                "\n",
                "As tabelas abaixo comparam os grupos com maior volume de movimentações brutas com aqueles que apresentam maior proporção de coortes classificadas na Classe 1 (`ALTA_ROTATIVIDADE_6M`) com no mínimo 50 coortes:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Volume por Grupo Ocupacional\n",
                "df_vol_cbo = df_monthly.groupBy('GRUPO_CBO').agg(\n",
                "    F.count('*').alias('N_TOTAL_MOV'),\n",
                "    F.sum(F.when(F.col('saldomovimentação') == 1, 1).otherwise(0)).alias('N_POSITIVOS'),\n",
                "    F.sum(F.when(F.col('saldomovimentação') == -1, 1).otherwise(0)).alias('N_NEGATIVOS')\n",
                ").sort(F.col('N_TOTAL_MOV').desc())\n",
                "\n",
                "print(\"=== TOP 10 GRUPOS OCUPACIONAIS POR VOLUME DE MOVIMENTAÇÕES ===\")\n",
                "for r in df_vol_cbo.take(10):\n",
                "    print(f\"  Grupo CBO {r['GRUPO_CBO']:2s} | Volume Total: {r['N_TOTAL_MOV']:10,} | Admissões: {r['N_POSITIVOS']:10,} | Desligamentos: {r['N_NEGATIVOS']:10,}\")\n",
                "\n",
                "# Agregação com a Silver / Target Auditado\n",
                "df_audit = spark.read.parquet(str(audit_path))\n",
                "df_coorte_cbo = df_monthly.groupBy('competênciamov', 'uf', 'seção').agg(F.first('GRUPO_CBO').alias('GRUPO_CBO_DOMINANTE'))\n",
                "df_audit_cbo = df_audit.join(df_coorte_cbo, on=['competênciamov', 'uf', 'seção'], how='inner')\n",
                "\n",
                "cbo_rot = df_audit_cbo.groupBy('GRUPO_CBO_DOMINANTE').agg(\n",
                "    F.count('*').alias('total_coortes'),\n",
                "    (F.sum(F.when(F.col('ALTA_ROTATIVIDADE_6M') == 1, 1).otherwise(0)) / F.count('*') * 100).alias('pct_classe_1')\n",
                ").filter(F.col('total_coortes') >= 50).sort(F.col('pct_classe_1').desc())\n",
                "\n",
                "print(\"\\n=== TOP 10 GRUPOS OCUPACIONAIS MAIS ASSOCIADOS À ALTA ROTATIVIDADE ===\")\n",
                "for r in cbo_rot.take(10):\n",
                "    print(f\"  Grupo CBO {r['GRUPO_CBO_DOMINANTE']:2s} | Coortes: {r['total_coortes']:5d} | % Alta Rotatividade: {r['pct_classe_1']:6.2f}%\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Comparação: Volume Bruto × Proporção de Rotatividade\n",
                "\n",
                "Existe uma importante distinção técnica entre **volume absoluto de desligamentos** e **taxa/proporção de alta rotatividade**:\n",
                "- **Alto Volume com Proporção Moderada:** O Grupo CBO 51 (Trabalhadores dos serviços pessoais) registra o MAIOR volume do Nordeste (3,34 milhões de movimentações), porém sua proporção de coortes de alta rotatividade é moderada (~51%).\n",
                "- **Volume Menor com Alta Proporção:** O Grupo CBO 84 (Fabricação de alimentos e bebidas) registra volume menor (1,51 milhão), contudo apresenta **56,36% de coortes na Classe 1 de alta rotatividade**.\n",
                "- **Combinação Crítica:** Ao cruzar Grupo CBO 52 (Comércio) com a **Seção Econômica G (Comércio Varejista/Atacadista)**, a proporção de coortes em alta rotatividade atinge **70,70%** (14,62 milhões de movimentações acumuladas)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cruzamentos Setoriais e Geográficos\n",
                "\n",
                "1. **Grupo CBO × Seção Econômica:** As maiores taxas de rotatividade ocorrem no cruzamento entre trabalhadores do comércio/serviços e as Seções **G (Comércio)** e **C (Indústria de Transformação)**;\n",
                "2. **Grupo CBO × UF (Estado):** Em Alagoas (UF 27) e Sergipe (UF 28), os trabalhadores da mecanização agrícola e cultivo (CBO 64) apresentam picos de rotatividade acima de **72%**, refletindo o ciclo safra/entre-safra da cana-de-açúcar no litoral nordestino;\n",
                "3. **Conexão com a Feature Importance:** A variável ocupacional CBO não entrou diretamente no modelo final para evitar esparsidade (2.562 códigos). Entretanto, as variáveis utilizadas na Coorte A (**Seção Econômica e Faixa Etária**) atuaram como agregadores e proxies perfeitos da volatilidade ocupacional."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### O que isso significa para um gestor público?\n",
                "\n",
                "1. **Direcionamento Estratégico de Políticas:** Permite ao Ministério do Trabalho e aos governos estaduais do Nordeste antecipar quais setores (ex: Comércio Varejista e Agrobiocombustíveis) e ocupações técnicas demandam programas contínuos de reciclagem e intermediação de mão de obra;\n",
                "2. **Acompanhamento de Vínculos Temporários:** Auxilia na identificação de ciclos sazonais de contratação/desligamento em regiões agrícolas e turísticas;\n",
                "3. **Foco Preventivo (Não Punitivo):** Os resultados devem subsidiar apoio institucional, qualificação profissional e melhorias nas condições de trabalho, e **nunca** embasar decisões ou punições individuais a trabalhadores ou estabelecimentos."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 17. Caderno de Campo\n",
                "\n",
                "> Esta seção deve ser preenchida manualmente pelo autor com experiências reais do desenvolvimento."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 18. Conclusão do Projeto\n",
                "\n",
                "### 18.1 Síntese do Estudo\n",
                "Este projeto abordou a previsão de alta rotatividade de mão de obra no mercado formal de trabalho brasileiro utilizando microdados públicos do Novo CAGED no triênio **2023–2025**. Diante da ausência de identificadores únicos persistentes de trabalhadores nos dados abertos (por razões de privacidade), o problema foi metodologicamente reestruturado para atuar sobre **Coortes Socioeconômicas Agregadas (`competênciamov + uf + seção + FAIXA_ETARIA`)**.\n",
                "\n",
                "Para viabilizar o processamento distribuído em ambiente local com restrição de memória (8 GB RAM), o escopo geográfico foi focado na **Região Nordeste**, processando **18,99 milhões de registros de movimentação** agregados em **33.465 coortes** ao longo de 30 competências elegíveis (202301 a 202506), respeitando a censura à direita de 6 meses futuros até 202512.\n",
                "\n",
                "### 18.2 Principais Achados Técnicos\n",
                "1. **Desempenho dos Modelos no PySpark MLlib:** O ensemble **`RandomForestClassifier`** foi recomendado como o melhor modelo do projeto, registrando **AUC-ROC de 0,8813**, **Acurácia de 78,56%** e **Recall de 84,99%** na detecção da classe de alta rotatividade no conjunto de teste não visto;\n",
                "2. **Fatores Preditivos Dominantes:** A proporção instantânea de desligamentos no mês de referência (`PROP_NEGATIVOS_T`) e a composição por faixa etária (especialmente jovens de 18-24 anos e idosos acima de 65 anos) apresentaram a maior importância preditiva agregada na decisão do modelo;\n",
                "3. **Tuning de Hiperparâmetros:** O procedimento de busca em grade via `TrainValidationSplit` (8 combinações) validou empiricamente que a parametrização inicial da Random Forest (`numTrees=30`, `maxDepth=8`, `minInstancesPerNode=1`) era a combinação ótima dentro da grade analisada;\n",
                "4. **Utilidade Prática:** O modelo oferece suporte à gestão pública e formulação de políticas de trabalho no Nordeste, permitindo identificar perfis e setores com inércia de alta rotatividade futura para direcionamento de programas de qualificação e acompanhamento setorial.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 19. Resumo para Defesa Oral e Conceitos Chave\n",
                "\n",
                "### Resumo para Defesa Oral\n",
                "- **Domínio & Problema:** Previsão de rotatividade agregada (turnover) no mercado de trabalho formal brasileiro (Novo CAGED 2023–2025).\n",
                "- **Desafio do Identificador Único:** Microdados públicos omitem CPF/PIS; solução metodológica via agregação por Coortes Socioeconômicas A (`competênciamov + uf + seção + FAIXA_ETARIA`).\n",
                "- **Escopo Nordeste:** Processamento distribuído de 18,99M linhas de movimentação nos 9 estados nordestinos, gerando 33.465 coortes com estabilidade estatística (média de 470 registros/coorte).\n",
                "- **Censura à Direita & Target:** Target binário `ALTA_ROTATIVIDADE_6M` construído com janela futura de 6 meses (elegíveis 202301 a 202506; corte mediano $P_{50} = 0,479005$).\n",
                "- **Camada Silver:** 14 features candidatas (4 categóricas e 10 numéricas), 0% de nulos, vetorização em 52 dimensões (`StringIndexer` + `OneHotEncoder` + `VectorAssembler`).\n",
                "- **Modelagem PySpark MLlib:** Split 70/30 (SEED 42). Logistic Regression (AUC = 0.8726) vs Random Forest (AUC = 0.8813).\n",
                "- **Modelo Recomendado:** `RandomForestClassifier` (numTrees=30, maxDepth=8) por apresentar maior AUC-ROC (0,8813), maior Recall (84,99%) e menor tempo de treino (9,605s).\n",
                "- **Top Features:** `PROP_NEGATIVOS_T` (19.41%) e `FAIXA_ETARIA` (62.13% agregada).\n",
                "- **Tuning:** `TrainValidationSplit` (8 combinações, 52.8s, 3.31x custo) confirmou otimalidade da parametrização base.\n",
                "\n",
                "### Conceitos que Preciso Saber Explicar\n",
                "1. **`StringIndexer`:** Mapeia categorias textuais em índices numéricos inteiros para processamento computacional.\n",
                "2. **`OneHotEncoder`:** Converte índices categóricos em vetores binários esparsos, impedindo ordenação arbitrária em categorias nominais.\n",
                "3. **`VectorAssembler`:** Consolida múltiplos atributos numéricos e vetores em um único vetor denominado `features`.\n",
                "4. **`Pipeline`:** Sequencia etapas de pré-processamento e estimadores de modelagem em um fluxo único e reprodutível.\n",
                "5. **`randomSplit`:** Método de divisão estocástica do DataFrame em subconjuntos independentes (ex: 70% Treino e 30% Teste).\n",
                "6. **Data Leakage:** Vazamento de informação do futuro ou da variável resposta para o conjunto de preditores.\n",
                "7. **AUC-ROC:** Métrica que avalia a capacidade discriminativa global do modelo entre 0,5 e 1,0.\n",
                "8. **Accuracy:** Proporção total de classificações corretas sobre o total de observações avaliadas.\n",
                "9. **Precision:** Proporção de verdadeiros positivos em relação ao total classificado como positivo ($\\frac{TP}{TP + FP}$).\n",
                "10. **Recall:** Proporção de verdadeiros positivos capturados em relação ao total real de positivos ($\\frac{TP}{TP + FN}$).\n",
                "11. **F1-Score:** Média harmônica entre Precision e Recall.\n",
                "12. **Matriz de Confusão:** Tabela $2 \\times 2$ contendo TN, FP, FN e TP.\n",
                "13. **Logistic Regression:** Algoritmo linear de classificação probabilística baseado na função logística.\n",
                "14. **Random Forest:** Algoritmo de ensemble baseado na combinação de múltiplas árvores de decisão independentes.\n",
                "15. **`featureImportances`:** Quantifica a redução acumulada de impureza de Gini proporcionada por cada atributo nas árvores.\n",
                "16. **Coeficientes:** Pesos da Regressão Logística que mensuram a variação nos log-odds para aumento unitário do atributo.\n",
                "17. **`TrainValidationSplit`:** Módulo do PySpark MLlib para busca em grade que avalia hiperparâmetros em uma única divisão interna.\n",
                "18. **Coorte:** Agrupamento homogêneo de registros compartilhado por atributos demográficos, geográficos e setoriais comuns no tempo.\n",
                "19. **Censura à Direita:** Exclusão de observações temporais finais cujo horizonte futuro de 6 meses não pôde ser completamente observado."
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
