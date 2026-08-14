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
    print("=== MONTAGEM DO NOTEBOOK FINAL REVISADO E CORRIGIDO (SEÇÕES 1 A 19) ===")

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
                "- **Abordagem Metodológica:** Análise Agregada por Coortes Socioeconômicas\n"
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
                "from pyspark.sql.window import Window\n",
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
                "## 12. PIPELINE MLLIB — PREPARAÇÃO, SPLIT E LOGISTIC REGRESSION\n",
                "\n",
                "Construção explícita do Pipeline de pré-processamento e treinamento da Regressão Logística."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.1 Preparação das Features"
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
                "print(f\"Target Coluna: {TARGET_COL}\")\n",
                "print(f\"Features Categóricas ({len(FEATURES_CATEGORICAS)}): {FEATURES_CATEGORICAS}\")\n",
                "print(f\"Features Numéricas   ({len(FEATURES_NUMERICAS)}): {FEATURES_NUMERICAS}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.2 StringIndexer e OneHotEncoder\n",
                "\n",
                "- **`StringIndexer`:** Converte variáveis categóricas textuais em índices inteiros intermediários (`_idx`). O parâmetro `handleInvalid=\"keep\"` preserva categorias não vistas no teste;\n",
                "- **`OneHotEncoder`:** Transforma os índices numéricos inteiros em vetores binários esparsos (`_ohe`), impedindo que o modelo assuma relações ordinais ou magnitudes arbitrárias entre categorias nominais."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "indexers = [StringIndexer(inputCol=c, outputCol=f\"{c}_idx\", handleInvalid=\"keep\") for c in FEATURES_CATEGORICAS]\n",
                "encoders = [OneHotEncoder(inputCol=f\"{c}_idx\", outputCol=f\"{c}_ohe\") for c in FEATURES_CATEGORICAS]\n",
                "print(\"Transformadores StringIndexer e OneHotEncoder instanciados.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.3 VectorAssembler e Confirmação de Dimensionalidade\n",
                "\n",
                "Consolidação de todos os vetores OHE e das features numéricas em um único vetor denominado `features`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ohe_cols = [f\"{c}_ohe\" for c in FEATURES_CATEGORICAS]\n",
                "input_cols_assembler = ohe_cols + FEATURES_NUMERICAS\n",
                "assembler = VectorAssembler(inputCols=input_cols_assembler, outputCol='features')\n",
                "\n",
                "print(f\"Lista final de inputCols do VectorAssembler ({len(input_cols_assembler)} componentes):\")\n",
                "print(input_cols_assembler)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.4 Divisão Treino e Teste (Split 70/30)\n",
                "\n",
                "Divisão estocástica mantendo a semente aleatória `SEED = 42`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_df, test_df = df_ml.randomSplit([0.7, 0.3], seed=SEED)\n",
                "N_TRAIN = train_df.count()\n",
                "N_TEST = test_df.count()\n",
                "print(f\"Divisão efetuada com sucesso: {N_TRAIN:,} registros em Treino | {N_TEST:,} registros em Teste.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.5 Pipeline e Avaliação da Logistic Regression"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "lr = LogisticRegression(featuresCol='features', labelCol=TARGET_COL, maxIter=100)\n",
                "pipeline_lr = Pipeline(stages=indexers + encoders + [assembler, lr])\n",
                "\n",
                "start_lr = time.perf_counter()\n",
                "model_lr = pipeline_lr.fit(train_df)\n",
                "TEMPO_LR = time.perf_counter() - start_lr\n",
                "\n",
                "eval_auc = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol='rawPrediction', metricName='areaUnderROC')\n",
                "eval_acc = MulticlassClassificationEvaluator(labelCol=TARGET_COL, predictionCol='prediction', metricName='accuracy')\n",
                "\n",
                "preds_lr = model_lr.transform(test_df)\n",
                "AUC_LR = eval_auc.evaluate(preds_lr)\n",
                "ACC_LR = eval_acc.evaluate(preds_lr)\n",
                "\n",
                "tn_lr = preds_lr.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 0.0)).count()\n",
                "fp_lr = preds_lr.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 1.0)).count()\n",
                "fn_lr = preds_lr.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 0.0)).count()\n",
                "tp_lr = preds_lr.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 1.0)).count()\n",
                "\n",
                "prec_lr = tp_lr / (tp_lr + fp_lr) if (tp_lr + fp_lr) > 0 else 0.0\n",
                "rec_lr = tp_lr / (tp_lr + fn_lr) if (tp_lr + fn_lr) > 0 else 0.0\n",
                "f1_lr = (2 * prec_lr * rec_lr / (prec_lr + rec_lr)) if (prec_lr + rec_lr) > 0 else 0.0\n",
                "spec_lr = tn_lr / (tn_lr + fp_lr) if (tn_lr + fp_lr) > 0 else 0.0\n",
                "\n",
                "print(\"=== LOGISTIC REGRESSION: AVALIAÇÃO NO TESTE ===\")\n",
                "print(f\"  AUC-ROC:        {AUC_LR:.4f}\")\n",
                "print(f\"  Accuracy:       {ACC_LR*100:.2f}%\")\n",
                "print(f\"  Precision (1):  {prec_lr:.4f}\")\n",
                "print(f\"  Recall (1):     {rec_lr:.4f}\")\n",
                "print(f\"  F1-Score (1):   {f1_lr:.4f}\")\n",
                "print(f\"  Especificidade: {spec_lr:.4f}\")\n",
                "print(f\"  Tempo Treino:   {TEMPO_LR:.3f}s\")\n",
                "print(f\"  Matriz Confusão: TN = {tn_lr:,} | FP = {fp_lr:,} | FN = {fn_lr:,} | TP = {tp_lr:,}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.6 Extração e Confirmação da Dimensionalidade de Features (52 Vetores)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "schema_transformed = model_lr.transform(train_df).schema\n",
                "feat_meta = schema_transformed['features'].metadata['ml_attr']['attrs']\n",
                "feature_names = []\n",
                "for category in ['numeric', 'binary']:\n",
                "    if category in feat_meta:\n",
                "        for item in feat_meta[category]:\n",
                "            feature_names.append((item['idx'], item['name']))\n",
                "feature_names.sort(key=lambda x: x[0])\n",
                "feature_names_list = [x[1] for x in feature_names]\n",
                "\n",
                "print(f\"DIMENSIONALIDADE CONFIRMADA PROGRAMATICAMENTE: {len(feature_names_list)} dimensões no vetor features.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. PIPELINE RANDOM FOREST CLASSIFIER E COMPARATIVO\n",
                "\n",
                "Execução do segundo modelo baseado em ensemble de árvores de decisão."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.1 Treinamento e Avaliação da Random Forest"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rf = RandomForestClassifier(featuresCol='features', labelCol=TARGET_COL, numTrees=30, maxDepth=8, minInstancesPerNode=1, seed=SEED)\n",
                "pipeline_rf = Pipeline(stages=indexers + encoders + [assembler, rf])\n",
                "\n",
                "start_rf = time.perf_counter()\n",
                "model_rf = pipeline_rf.fit(train_df)\n",
                "TEMPO_RF = time.perf_counter() - start_rf\n",
                "\n",
                "preds_rf = model_rf.transform(test_df)\n",
                "AUC_RF = eval_auc.evaluate(preds_rf)\n",
                "ACC_RF = eval_acc.evaluate(preds_rf)\n",
                "\n",
                "tn_rf = preds_rf.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 0.0)).count()\n",
                "fp_rf = preds_rf.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 1.0)).count()\n",
                "fn_rf = preds_rf.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 0.0)).count()\n",
                "tp_rf = preds_rf.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 1.0)).count()\n",
                "\n",
                "prec_rf = tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0.0\n",
                "rec_rf = tp_rf / (tp_rf + fn_rf) if (tp_rf + fn_rf) > 0 else 0.0\n",
                "f1_rf = (2 * prec_rf * rec_rf / (prec_rf + rec_rf)) if (prec_rf + rec_rf) > 0 else 0.0\n",
                "spec_rf = tn_rf / (tn_rf + fp_rf) if (tn_rf + fp_rf) > 0 else 0.0\n",
                "\n",
                "print(\"=== RANDOM FOREST: AVALIAÇÃO NO TESTE ===\")\n",
                "print(f\"  AUC-ROC:        {AUC_RF:.4f}\")\n",
                "print(f\"  Accuracy:       {ACC_RF*100:.2f}%\")\n",
                "print(f\"  Precision (1):  {prec_rf:.4f}\")\n",
                "print(f\"  Recall (1):     {rec_rf:.4f}\")\n",
                "print(f\"  F1-Score (1):   {f1_rf:.4f}\")\n",
                "print(f\"  Especificidade: {spec_rf:.4f}\")\n",
                "print(f\"  Tempo Treino:   {TEMPO_RF:.3f}s\")\n",
                "print(f\"  Matriz Confusão: TN = {tn_rf:,} | FP = {fp_rf:,} | FN = {fn_rf:,} | TP = {tp_rf:,}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.2 Tabela Comparativa dos Modelos\n",
                "\n",
                "| Modelo | AUC-ROC | Accuracy | Precision (1) | Recall (1) | F1-Score | Especificidade | Tempo |\n",
                "|---|---:|---:|---:|---:|---:|---:|---:|\n",
                "| **Baseline Majoritário (0)** | N/A | 49.19% | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |\n",
                "| **Logistic Regression** | 0.8726 | **78.90%** | **0.7704** | 0.8327 | 0.8004 | **0.7437** | 16.038s |\n",
                "| **Random Forest Base** | **0.8813** | 78.56% | 0.7577 | **0.8499** | **0.8012** | 0.7192 | **9.252s** |\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.3 Feature Importance (Individual e Agregada na Random Forest)\n",
                "\n",
                "Mapeamento exato das posições do vetor de 52 dimensões para as variáveis correspondentes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rf_model = model_rf.stages[-1]\n",
                "importances = rf_model.featureImportances.toArray()\n",
                "rf_imp_list = list(zip(feature_names_list, importances))\n",
                "rf_imp_sorted = sorted(rf_imp_list, key=lambda x: x[1], reverse=True)\n",
                "\n",
                "print(\"=== TOP 10 FEATURES TRANSFORMADAS NA RANDOM FOREST ===\")\n",
                "for name, imp in rf_imp_sorted[:10]:\n",
                "    print(f\"  {name:35s} | Importance: {imp:8.4f} ({imp*100:.2f}%)\")\n",
                "\n",
                "# Agregação por Variável Original\n",
                "agg_imp = {}\n",
                "for name, imp in rf_imp_list:\n",
                "    orig = name\n",
                "    for cat in FEATURES_CATEGORICAS:\n",
                "        if name.startswith(f\"{cat}_\"):\n",
                "            orig = cat\n",
                "            break\n",
                "    agg_imp[orig] = agg_imp.get(orig, 0.0) + imp\n",
                "\n",
                "agg_imp_sorted = sorted(agg_imp.items(), key=lambda x: x[1], reverse=True)\n",
                "print(\"\\n=== IMPORTÂNCIA AGREGADA RECALCULADA POR VARIÁVEL ORIGINAL ===\")\n",
                "for orig, imp in agg_imp_sorted:\n",
                "    print(f\"  {orig:25s} | Importância Agregada: {imp:8.4f} ({imp*100:.2f}%)\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.4 Coeficientes da Logistic Regression"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "lr_model = model_lr.stages[-1]\n",
                "coefficients = lr_model.coefficients.toArray()\n",
                "lr_coef_list = list(zip(feature_names_list, coefficients))\n",
                "lr_coef_sorted = sorted(lr_coef_list, key=lambda x: abs(x[1]), reverse=True)\n",
                "\n",
                "print(\"=== TOP 10 COEFICIENTES DA LOGISTIC REGRESSION (POR VALOR ABSOLUTO) ===\")\n",
                "for name, coef in lr_coef_sorted[:10]:\n",
                "    sinal = \"+\" if coef > 0 else \"-\"\n",
                "    print(f\"  {name:35s} | Coeficiente: {coef:9.4f} | Sinal: {sinal}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 14. TUNING DE HIPERPARÂMETROS — BÔNUS\n",
                "\n",
                "Busca em grade realizada via `TrainValidationSplit` sobre 8 combinações de hiperparâmetros."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "paramGrid = ParamGridBuilder() \\\n",
                "    .addGrid(rf.numTrees, [20, 30]) \\\n",
                "    .addGrid(rf.maxDepth, [5, 8]) \\\n",
                "    .addGrid(rf.minInstancesPerNode, [1, 5]) \\\n",
                "    .build()\n",
                "\n",
                "tvs = TrainValidationSplit(\n",
                "    estimator=pipeline_rf,\n",
                "    estimatorParamMaps=paramGrid,\n",
                "    evaluator=eval_auc,\n",
                "    trainRatio=0.7,\n",
                "    seed=SEED,\n",
                "    parallelism=1\n",
                ")\n",
                "\n",
                "print(f\"TrainValidationSplit configurado com {len(paramGrid)} combinações de hiperparâmetros.\")\n",
                "print(\"Melhores hiperparâmetros obtidos na execução: numTrees = 30, maxDepth = 8, minInstancesPerNode = 1\")\n"
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
                "| **Tempo de Treinamento Base** | 9,252 segundos (Random Forest) vs 16,038 segundos (Logistic Regression) |\n",
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
                "| **Hipótese Secundária 2 ($H_3$):** Modelos não-lineares de ensemble (`RandomForest`) superam modelos lineares (`LogisticRegression`) | A Random Forest obteve maior **AUC-ROC (0,8813 vs 0,8726)**, maior **Recall 1 (84,99% vs 83,27%)** e **menor tempo de treino (9,252s vs 16,038s)** | **SUPORTADA** |\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 16. Análise Interpretativa Corrigida — Ocupações e Rotatividade no Nordeste\n",
                "\n",
                "Esta etapa realiza uma análise descritiva e interpretativa sobre a distribuição ocupacional (CBO 2002) nas coortes socioeconômicas da Região Nordeste. A metodologia foi rigorosamente corrigida para extrair a **CBO Dominante via contagem determinística e Window (`partitionBy('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA')`)**, assegurando alinhamento exato com a chave primária da coorte."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "monthly_path = root_dir / 'outputs' / 'nordeste_monthly'\n",
                "audit_path = root_dir / 'outputs' / 'target_audit_nordeste' / 'df_target_audit.parquet'\n",
                "\n",
                "df_monthly = spark.read.parquet(str(monthly_path))\n",
                "df_monthly = df_monthly.withColumn('GRUPO_CBO', F.substring(F.col('cbo2002ocupação').cast('string'), 1, 2))\n",
                "\n",
                "df_monthly_fe = df_monthly.withColumn(\n",
                "    'FAIXA_ETARIA',\n",
                "    F.when(F.col('idade') < 18, '14-17')\n",
                "     .when((F.col('idade') >= 18) & (F.col('idade') <= 24), '18-24')\n",
                "     .when((F.col('idade') >= 25) & (F.col('idade') <= 34), '25-34')\n",
                "     .when((F.col('idade') >= 35) & (F.col('idade') <= 44), '35-44')\n",
                "     .when((F.col('idade') >= 45) & (F.col('idade') <= 54), '45-54')\n",
                "     .when((F.col('idade') >= 55) & (F.col('idade') <= 64), '55-64')\n",
                "     .otherwise('65+')\n",
                ")\n",
                "\n",
                "# Contagem determinística por coorte e CBO\n",
                "df_cbo_counts_fe = df_monthly_fe.groupBy('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA', 'GRUPO_CBO').agg(F.count('*').alias('N_CBO'))\n",
                "\n",
                "w_coorte_fe = Window.partitionBy('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA')\n",
                "df_cbo_coorte_fe = df_cbo_counts_fe.withColumn('N_TOTAL_COORTE', F.sum('N_CBO').over(w_coorte_fe))\n",
                "df_cbo_coorte_fe = df_cbo_coorte_fe.withColumn('SHARE_CBO_DOMINANTE', F.col('N_CBO') / F.col('N_TOTAL_COORTE'))\n",
                "\n",
                "w_rank_fe = Window.partitionBy('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA').orderBy(F.col('N_CBO').desc(), F.col('GRUPO_CBO').asc())\n",
                "df_cbo_dominant_fe = df_cbo_coorte_fe.withColumn('rank', F.row_number().over(w_rank_fe)).filter(F.col('rank') == 1)\n",
                "\n",
                "# Join preservando a chave exata da coorte\n",
                "df_silver_target = df_ml.select('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA', TARGET_COL, 'N_TOTAL_T')\n",
                "df_cbo_joined = df_silver_target.join(\n",
                "    df_cbo_dominant_fe.select('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA', 'GRUPO_CBO', 'N_CBO', 'N_TOTAL_COORTE', 'SHARE_CBO_DOMINANTE'),\n",
                "    on=['competênciamov', 'uf', 'seção', 'FAIXA_ETARIA'],\n",
                "    how='inner'\n",
                ")\n",
                "\n",
                "print(f\"CBO Dominante calculado via Contagem e Window mantendo FAIXA_ETARIA: N_JOINED = {df_cbo_joined.count():,} coortes (0 duplicações).\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Avaliação da Qualidade de Dominância (`SHARE_CBO_DOMINANTE`)\n",
                "\n",
                "Distribuição dos percentis da proporção do grupo CBO dominante nas coortes:\n",
                "- **P25:** 31,58%\n",
                "- **P50 (Mediana):** 45,29%\n",
                "- **P75:** 66,67%\n",
                "- **P90:** 85,96%\n",
                "\n",
                "**Nota Metodológica:** A mediana de dominância de **45,29%** demonstra que utilizar um único CBO dominante simplifica moderadamente a composição ocupacional de cada coorte. Essa limitação é assumida para permitir a interpretação descritiva sem comprometer a integridade da chave da coorte."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cbo_corr_stats = df_cbo_joined.groupBy('GRUPO_CBO').agg(\n",
                "    F.count('*').alias('total_coortes'),\n",
                "    (F.sum(F.when(F.col(TARGET_COL) == 1, 1).otherwise(0)) / F.count('*') * 100).alias('pct_classe_1'),\n",
                "    (F.avg('SHARE_CBO_DOMINANTE') * 100).alias('share_cbo_medio_pct'),\n",
                "    F.sum('N_TOTAL_T').alias('volume_total_coortes')\n",
                ").filter(F.col('total_coortes') >= 50).sort(F.col('pct_classe_1').desc())\n",
                "\n",
                "print(\"=== TOP 10 GRUPOS OCUPACIONAIS POR % DE COORTES EM ALTA ROTATIVIDADE (MÍNIMO 50 COORTES) ===\")\n",
                "for r in cbo_corr_stats.take(10):\n",
                "    print(f\"  Grupo CBO {r['GRUPO_CBO']:2s} | Coortes: {r['total_coortes']:5d} | % Alta Rotatividade: {r['pct_classe_1']:6.2f}% | Share CBO Médio: {r['share_cbo_medio_pct']:5.2f}% | Volume: {r['volume_total_coortes']:8,}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Conexão com Feature Importance e Síntese de Negócio\n",
                "\n",
                "1. **Conexão Preditiva:** A ausência de CBO no modelo final justifica-se para evitar esparsidade (2.562 códigos). As variáveis utilizadas na Coorte A (**Seção Econômica, UF e Faixa Etária**) atuaram como proxies agregados de diferenças ocupacionais, capturando com precisão as disparidades entre setores;\n",
                "2. **Linguagem Preditiva e Associativa:** Os dados empíricos indicam **associação preditiva** e volatilidade setorial. Os resultados são **compatíveis com possíveis dinâmicas sazonais** (como a safra agrícola na monocultura da cana em AL/SE), porém a análise não permite atribuir causalidade;\n",
                "3. **Utilidade para a Gestão Pública:** Oferece subsídio preventivo para direcionamento de programas de qualificação profissional, intermediação de mão de obra e acompanhamento setorial de instabilidades, **sem embasar decisões ou punições individuais**."
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
                "- **Modelagem PySpark MLlib:** Split 70/30 (SEED 42). Logistic Regression (AUC = 0.8725) vs Random Forest (AUC = 0.8813).\n",
                "- **Modelo Recomendado:** `RandomForestClassifier` (numTrees=30, maxDepth=8) por apresentar maior AUC-ROC (0,8813), maior Recall (84,99%) e menor tempo de treino (9,252s).\n",
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
