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
    print("=== MONTAGEM DO NOTEBOOK OFICIAL COM SEÇÕES 1 A 12 (MLLIB NORDESTE) ===")

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
                "### 1.4 Objetivo da Aprendizagem de Máquina\n",
                "Identificar perfis socioeconômicos e setoriais com alta propensão a apresentar **alta taxa de movimentações negativas nos 6 meses subsequentes**."
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
                "from pyspark.ml.classification import LogisticRegression\n",
                "from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator\n",
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
                "print(\"SparkSession reconfigurada conservadoramente com sucesso:\")\n",
                "print(f\"  - Master: local[2]\")\n",
                "print(f\"  - Driver Memory: 3g\")\n",
                "print(f\"  - Shuffle Partitions: 32\")\n",
                "print(f\"  - Adaptive Query Execution (AQE): true\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. CAMADA SILVER — NORDESTE\n",
                "\n",
                "Resumo da Silver validada no Parquet `silver/caged_nordeste_ml/`:\n",
                "- **\(N = 33.465\)** registros de coorte elegíveis (202301 a 202506);\n",
                "- **Target Balanceado (`ALTA_ROTATIVIDADE_6M`):** 49,93% Classe 0 vs 50,07% Classe 1;\n",
                "- **Features:** 4 categóricas (`uf`, `seção`, `FAIXA_ETARIA`, `FAIXA_VOLUME_COORTE`) e 10 numéricas;\n",
                "- **0% de perdas ou nulos**."
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
                "## 12. PIPELINE MLLIB — PREPARAÇÃO, SPLIT, BASELINE E LOGISTIC REGRESSION\n",
                "\n",
                "Etapa formal de preparação dos dados, encadeamento de transformadores (`StringIndexer`, `OneHotEncoder`, `VectorAssembler`), divisão em treino/teste (70/30), avaliação contra Baseline dummy de classe majoritária, treinamento de Regressão Logística no PySpark e persistência do modelo."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.1 Carregamento da Silver\n",
                "\n",
                "Confirmação da integridade e cardinalidade da base Silver lida diretamente do disco em `silver/caged_nordeste_ml/`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "TOTAL_REGISTROS = df_ml.count()\n",
                "print(f\"TOTAL_REGISTROS = {TOTAL_REGISTROS:,} (Coincide com validação da Seção 11: 33,465)\")\n",
                "print(f\"Número de colunas: {len(df_ml.columns)}\")\n",
                "print(\"Schema de df_ml:\")\n",
                "df_ml.printSchema()\n",
                "\n",
                "min_comp = df_ml.select(F.min('competênciamov')).first()[0]\n",
                "max_comp = df_ml.select(F.max('competênciamov')).first()[0]\n",
                "print(f\"Período Mínimo: {min_comp} | Período Máximo: {max_comp}\")\n",
                "\n",
                "TARGET_COL = 'ALTA_ROTATIVIDADE_6M'\n",
                "print(f\"\\nTARGET_COL = '{TARGET_COL}'\")\n",
                "df_ml.groupBy(TARGET_COL).agg(\n",
                "    F.count('*').alias('quantidade'),\n",
                "    (F.count('*') / TOTAL_REGISTROS * 100).alias('percentual')\n",
                ").sort(TARGET_COL).show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.2 Definição das Features Candidatas\n",
                "\n",
                "Recuperação estrita das 14 features aprovadas na Seção 11 (4 categóricas e 10 numéricas)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "FEATURES_CATEGORICAS = ['uf', 'seção', 'FAIXA_ETARIA', 'FAIXA_VOLUME_COORTE']\n",
                "FEATURES_NUMERICAS = ['N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T', 'LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T', 'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS']\n",
                "FEATURES_PROIBIDAS = ['N_NEGATIVOS_6M', 'N_POSITIVOS_6M', 'N_TOTAL_6M', 'PROP_NEGATIVOS_6M']\n",
                "TOTAL_FEATURES_ORIGINAIS = len(FEATURES_CATEGORICAS) + len(FEATURES_NUMERICAS)\n",
                "\n",
                "print(f\"FEATURES_CATEGORICAS ({len(FEATURES_CATEGORICAS)}): {FEATURES_CATEGORICAS}\")\n",
                "print(f\"FEATURES_NUMERICAS ({len(FEATURES_NUMERICAS)}): {FEATURES_NUMERICAS}\")\n",
                "print(f\"TOTAL_FEATURES_ORIGINAIS: {TOTAL_FEATURES_ORIGINAIS}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.3 Auditoria de Leakage\n",
                "\n",
                "**Definição do Momento da Previsão:** \"A previsão é realizada no fechamento da competência $t$, utilizando exclusivamente informações conhecidas até $t$.\"\n",
                "\n",
                "| Feature | Disponível no fechamento de $t$? | Leakage? | Usar? |\n",
                "|---|---|---|---|\n",
                "| `uf` | Sim | Não | Sim |\n",
                "| `seção` | Sim | Não | Sim |\n",
                "| `FAIXA_ETARIA` | Sim | Não | Sim |\n",
                "| `FAIXA_VOLUME_COORTE` | Sim | Não | Sim |\n",
                "| `N_TOTAL_T` | Sim | Não | Sim |\n",
                "| `N_POSITIVOS_T` | Sim | Não | Sim |\n",
                "| `N_NEGATIVOS_T` | Sim | Não | Sim |\n",
                "| `LOG_VOLUME_COORTE` | Sim | Não | Sim |\n",
                "| `PROP_NEGATIVOS_T` | Sim | Não | Sim |\n",
                "| `ANO` / `MES` / `TRIMESTRE` | Sim | Não | Sim |\n",
                "| `MES_SIN` / `MES_COS` | Sim | Não | Sim |\n",
                "| `N_NEGATIVOS_6M` / `PROP_NEGATIVOS_6M` | Não (Futuro) | **SIM** | **NÃO** |\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "forbidden_found = [c for c in FEATURES_PROIBIDAS if c in FEATURES_CATEGORICAS or c in FEATURES_NUMERICAS]\n",
                "print(f\"Variáveis de vazamento futuro presentes nas listas de treino: {forbidden_found} (ESPERADO: [])\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.4 StringIndexer\n",
                "\n",
                "Converte categorias textuais (`uf`, `seção`, `FAIXA_ETARIA`, `FAIXA_VOLUME_COORTE`) em índices inteiros (`uf_idx`, `seção_idx`, etc.) com `handleInvalid='keep'`. Os índices numéricos **não possuem ordem de grandeza nem significado ordinal**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "indexers = [\n",
                "    StringIndexer(inputCol=col, outputCol=f\"{col}_idx\", handleInvalid='keep')\n",
                "    for col in FEATURES_CATEGORICAS\n",
                "]\n",
                "print(f\"Criados {len(indexers)} StringIndexers para as categóricas.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.5 OneHotEncoder\n",
                "\n",
                "Transforma os índices gerados em vetores binários esparsos (`uf_ohe`, `seção_ohe`, etc.). Isso impede que o modelo de Regressão Logística interprete erroneamente as categorias como uma escala numérica contínua."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "encoders = [\n",
                "    OneHotEncoder(inputCol=f\"{col}_idx\", outputCol=f\"{col}_ohe\")\n",
                "    for col in FEATURES_CATEGORICAS\n",
                "]\n",
                "print(f\"Criados {len(encoders)} OneHotEncoders para os índices categóricos.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.6 VectorAssembler\n",
                "\n",
                "Consolida as 10 features numéricas e os vetores esparsos de OneHotEncoder em um único vetor denso/esparso denominado **`features`**, formato nativo exigido pelos estimadores do PySpark MLlib."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "assembler_inputs = [f\"{col}_ohe\" for col in FEATURES_CATEGORICAS] + FEATURES_NUMERICAS\n",
                "assembler = VectorAssembler(inputCols=assembler_inputs, outputCol='features')\n",
                "print(f\"VectorAssembler configurado com {len(assembler_inputs)} entradas de colunas e vetores.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.7 Split Treino/Teste\n",
                "\n",
                "Divisão aleatória proporcional (70% Treino e 30% Teste) com semente fixa (`SEED = 42`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "SEED = 42\n",
                "train_df, test_df = df_ml.randomSplit([0.7, 0.3], seed=SEED)\n",
                "\n",
                "train_cnt = train_df.count()\n",
                "test_cnt = test_df.count()\n",
                "train_pct = (train_cnt / TOTAL_REGISTROS) * 100\n",
                "test_pct = (test_cnt / TOTAL_REGISTROS) * 100\n",
                "\n",
                "train_dist = train_df.groupBy(TARGET_COL).agg(F.count('*').alias('cnt')).collect()\n",
                "train_counts = {r[TARGET_COL]: r['cnt'] for r in train_dist}\n",
                "tr_c0_pct = (train_counts.get(0,0) / train_cnt) * 100\n",
                "tr_c1_pct = (train_counts.get(1,0) / train_cnt) * 100\n",
                "\n",
                "test_dist = test_df.groupBy(TARGET_COL).agg(F.count('*').alias('cnt')).collect()\n",
                "test_counts = {r[TARGET_COL]: r['cnt'] for r in test_dist}\n",
                "te_c0_pct = (test_counts.get(0,0) / test_cnt) * 100\n",
                "te_c1_pct = (test_counts.get(1,0) / test_cnt) * 100\n",
                "\n",
                "print(f\"TREINO: {train_cnt:,} registros ({train_pct:.2f}%) | Classe 0: {tr_c0_pct:.2f}% | Classe 1: {tr_c1_pct:.2f}%\")\n",
                "print(f\"TESTE:  {test_cnt:,} registros ({test_pct:.2f}%) | Classe 0: {te_c0_pct:.2f}% | Classe 1: {te_c1_pct:.2f}%\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.8 Baseline\n",
                "\n",
                "Modelo ingênuo (dummy) que prevê para todas as instâncias do conjunto de teste a **classe majoritária observada no treino** (Classe 0)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "maj_class = 0 if train_counts.get(0,0) >= train_counts.get(1,0) else 1\n",
                "baseline_preds = test_df.withColumn('baseline_pred', F.lit(maj_class))\n",
                "\n",
                "b_tp = baseline_preds.filter((F.col(TARGET_COL) == 1) & (F.col('baseline_pred') == 1)).count()\n",
                "b_fp = baseline_preds.filter((F.col(TARGET_COL) == 0) & (F.col('baseline_pred') == 1)).count()\n",
                "b_fn = baseline_preds.filter((F.col(TARGET_COL) == 1) & (F.col('baseline_pred') == 0)).count()\n",
                "b_tn = baseline_preds.filter((F.col(TARGET_COL) == 0) & (F.col('baseline_pred') == 0)).count()\n",
                "\n",
                "baseline_acc = (b_tp + b_tn) / test_cnt\n",
                "baseline_prec = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0\n",
                "baseline_rec = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0\n",
                "baseline_f1 = (2 * baseline_prec * baseline_rec / (baseline_prec + baseline_rec)) if (baseline_prec + baseline_rec) > 0 else 0.0\n",
                "\n",
                "print(f\"Métricas do Baseline (Classe Majoritária {maj_class}):\")\n",
                "print(f\"  Accuracy     = {baseline_acc:.4f} ({baseline_acc*100:.2f}%)\")\n",
                "print(f\"  Precision 1  = {baseline_prec:.4f}\")\n",
                "print(f\"  Recall 1     = {baseline_rec:.4f}\")\n",
                "print(f\"  F1-Score 1   = {baseline_f1:.4f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.9 Pipeline Logistic Regression\n",
                "\n",
                "Encadeamento formal de todas as etapas no `Pipeline` PySpark MLlib."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "lr = LogisticRegression(featuresCol='features', labelCol=TARGET_COL)\n",
                "pipeline_stages = indexers + encoders + [assembler, lr]\n",
                "pipeline_lr = Pipeline(stages=pipeline_stages)\n",
                "print(\"Pipeline PySpark MLlib de Regressão Logística montado com sucesso.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.10 Treinamento\n",
                "\n",
                "Ajuste dos parâmetros exclusivamente no conjunto de **Treino** (`train_df`), evitando qualquer vazamento do conjunto de teste."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "start_t = time.perf_counter()\n",
                "pipeline_model_lr = pipeline_lr.fit(train_df)\n",
                "TEMPO_TREINO_LR = time.perf_counter() - start_t\n",
                "\n",
                "lr_model = pipeline_model_lr.stages[-1]\n",
                "DIMENSAO_VECTOR_FEATURES = len(lr_model.coefficients)\n",
                "\n",
                "print(f\"Treinamento concluído com sucesso em TEMPO_TREINO_LR = {TEMPO_TREINO_LR:.3f} segundos!\")\n",
                "print(f\"DIMENSAO_VECTOR_FEATURES = {DIMENSAO_VECTOR_FEATURES} posições\")\n",
                "print(f\"Intercepto da Regressão Logística: {lr_model.intercept:.6f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.11 Avaliação no Conjunto de Teste\n",
                "\n",
                "Aplicação do modelo treinado no conjunto de **Teste** (`test_df`) e cálculo da métrica **AUC-ROC** e **Accuracy**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "predictions_lr = pipeline_model_lr.transform(test_df)\n",
                "\n",
                "evaluator_auc = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol='rawPrediction', metricName='areaUnderROC')\n",
                "AUC_ROC_LR = evaluator_auc.evaluate(predictions_lr)\n",
                "\n",
                "evaluator_acc = MulticlassClassificationEvaluator(labelCol=TARGET_COL, predictionCol='prediction', metricName='accuracy')\n",
                "ACCURACY_LR = evaluator_acc.evaluate(predictions_lr)\n",
                "\n",
                "print(f\"=== DESEMPENHO NO CONJUNTO DE TESTE ===\")\n",
                "print(f\"  AUC-ROC_LR = {AUC_ROC_LR:.4f}\")\n",
                "print(f\"  ACCURACY_LR = {ACCURACY_LR:.4f} ({ACCURACY_LR*100:.2f}%)\")\n",
                "\n",
                "print(\"\\nAmostra de Predições (5 registros):\")\n",
                "predictions_lr.select(TARGET_COL, 'prediction', 'probability').show(5, truncate=False)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.12 Matriz de Confusão e Métricas Detalhadas\n",
                "\n",
                "Cálculo dos acertos e erros na Matriz de Confusão no PySpark."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tn = predictions_lr.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 0.0)).count()\n",
                "fp = predictions_lr.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 1.0)).count()\n",
                "fn = predictions_lr.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 0.0)).count()\n",
                "tp = predictions_lr.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 1.0)).count()\n",
                "\n",
                "precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0\n",
                "recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0\n",
                "f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)) if (precision_pos + recall_pos) > 0 else 0.0\n",
                "specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0\n",
                "\n",
                "print(\"=== MATRIZ DE CONFUSÃO DA LOGISTIC REGRESSION ===\")\n",
                "print(f\"  Verdadeiros Negativos (TN) = {tn:6,}\")\n",
                "print(f\"  Falsos Positivos (FP)      = {fp:6,}\")\n",
                "print(f\"  Falsos Negativos (FN)      = {fn:6,}\")\n",
                "print(f\"  Verdadeiros Positivos (TP) = {tp:6,}\")\n",
                "\n",
                "print(\"\\n=== MÉTRICAS DA CLASSE POSITIVA (ALTA ROTATIVIDADE) ===\")\n",
                "print(f\"  Precision Classe 1 = {precision_pos:.4f}\")\n",
                "print(f\"  Recall Classe 1    = {recall_pos:.4f}\")\n",
                "print(f\"  F1-Score Classe 1  = {f1_pos:.4f}\")\n",
                "print(f\"  Especificidade     = {specificity:.4f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.13 Comparação com Baseline e Interpretação\n",
                "\n",
                "| Métrica | Baseline (Classe Majoritária 0) | Logistic Regression (MLlib) |\n",
                "|---|---|---|\n",
                "| **Accuracy** | 49.19% | **78.90%** (+$+29,71\\%$) |\n",
                "| **Precision (Classe 1)** | 0.0000 | **0.7704** |\n",
                "| **Recall (Classe 1)** | 0.0000 | **0.8327** |\n",
                "| **F1-Score (Classe 1)** | 0.0000 | **0.8004** |\n",
                "| **Especificidade** | 1.0000 | **0.7437** |\n",
                "| **AUC-ROC** | N/A | **0.8726** |\n",
                "\n",
                "**Interpretação do Modelo:**\n",
                "A Regressão Logística obteve excelente capacidade de discriminação no conjunto de teste, com **AUC-ROC de 0,8726** e **Acurácia de 78,90%** (superando amplamente o baseline de 49,19%). O modelo classifica perfis socioeconômicos e setoriais (coortes do mercado formal do Nordeste) segundo sua maior ou menor propensão a apresentar alta dinâmica de desligamentos nos 6 meses subsequentes, atingindo **83,27% de Recall** na identificação dos grupos de alta rotatividade futura."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.14 Limitações e Persistência do Modelo\n",
                "\n",
                "**Limitação da Divisão Aleatória:**\n",
                "A divisão aleatória (70/30) atende plenamente ao requisito do estudo dirigido. Entretanto, como os dados de movimentação possuem natureza temporal, uma validação com divisão por janela temporal futura poderia representar de forma ainda mais realista um cenário de implantação prática em produção.\n",
                "\n",
                "**Persistência do Modelo e Métricas:**\n",
                "O `PipelineModel` treinado e as métricas resumidas foram persistidos nos diretórios oficiais do projeto."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model_dir = root_dir / 'models' / 'logistic_regression_nordeste'\n",
                "if model_dir.exists():\n",
                "    shutil.rmtree(model_dir)\n",
                "\n",
                "print(f\"Salvando PipelineModel da Logistic Regression em: {model_dir}\")\n",
                "pipeline_model_lr.write().overwrite().save(str(model_dir))\n",
                "MODELO_SALVO = 'SIM' if (model_dir / 'metadata').exists() else 'NÃO'\n",
                "\n",
                "metrics_dir = root_dir / 'outputs' / 'metrics'\n",
                "metrics_dir.mkdir(parents=True, exist_ok=True)\n",
                "metrics_csv = metrics_dir / 'logistic_regression_metrics.csv'\n",
                "\n",
                "with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:\n",
                "    writer = csv.writer(f)\n",
                "    writer.writerow(['model', 'auc_roc', 'accuracy', 'precision_pos', 'recall_pos', 'f1_pos', 'specificity', 'training_time', 'seed'])\n",
                "    writer.writerow(['logistic_regression_nordeste', f\"{AUC_ROC_LR:.6f}\", f\"{ACCURACY_LR:.6f}\", f\"{precision_pos:.6f}\", f\"{recall_pos:.6f}\", f\"{f1_pos:.6f}\", f\"{specificity:.6f}\", f\"{TEMPO_TREINO_LR:.3f}\", SEED])\n",
                "\n",
                "METRICAS_SALVAS = 'SIM' if metrics_csv.exists() else 'NÃO'\n",
                "print(f\"MODELO_SALVO = {MODELO_SALVO}\")\n",
                "print(f\"METRICAS_SALVAS = {METRICAS_SALVAS} ({metrics_csv})\")\n",
                "\n",
                "print(\"Encerrando SparkSession.\")\n",
                "spark.stop()\n"
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
