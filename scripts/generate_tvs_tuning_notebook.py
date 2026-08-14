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
    print("=== MONTAGEM DO NOTEBOOK OFICIAL COM SEÇÕES 1 A 14 (TUNING NORDESTE) ===")

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
                "## 12 e 13. MODELOS DE REFERÊNCIA — LOGISTIC REGRESSION & RANDOM FOREST\n",
                "\n",
                "Reutilização estrita das 14 features e do mesmo split de Treino (23.510) e Teste (9.955) com `SEED = 42`.\n",
                "- **Logistic Regression:** AUC-ROC = 0.8726 | Accuracy = 78.90% | Tempo = 15.465s\n",
                "- **Random Forest Base:** AUC-ROC = 0.8813 | Accuracy = 78.56% | Tempo = 9.605s\n",
                "- **Modelo Escolhido para Tuning:** `RandomForestClassifier` por apresentar maior AUC-ROC e maior Recall na classe de alta rotatividade."
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
                "\n",
                "indexers = [StringIndexer(inputCol=c, outputCol=f\"{c}_idx\", handleInvalid='keep') for c in FEATURES_CATEGORICAS]\n",
                "encoders = [OneHotEncoder(inputCol=f\"{c}_idx\", outputCol=f\"{c}_ohe\") for c in FEATURES_CATEGORICAS]\n",
                "assembler = VectorAssembler(inputCols=[f\"{c}_ohe\" for c in FEATURES_CATEGORICAS] + FEATURES_NUMERICAS, outputCol='features')\n",
                "\n",
                "rf_base = RandomForestClassifier(featuresCol='features', labelCol=TARGET_COL, numTrees=30, maxDepth=8, seed=SEED)\n",
                "pipeline_base = Pipeline(stages=indexers + encoders + [assembler, rf_base])\n",
                "\n",
                "start_base = time.perf_counter()\n",
                "model_base = pipeline_base.fit(train_df)\n",
                "TEMPO_TREINO_BASE = time.perf_counter() - start_base\n",
                "\n",
                "eval_auc = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol='rawPrediction', metricName='areaUnderROC')\n",
                "eval_acc = MulticlassClassificationEvaluator(labelCol=TARGET_COL, predictionCol='prediction', metricName='accuracy')\n",
                "\n",
                "preds_base = model_base.transform(test_df)\n",
                "AUC_ROC_BASE = eval_auc.evaluate(preds_base)\n",
                "ACCURACY_BASE = eval_acc.evaluate(preds_base)\n",
                "\n",
                "print(f\"Random Forest Base re-executada: AUC-ROC = {AUC_ROC_BASE:.4f} | Tempo Base = {TEMPO_TREINO_BASE:.3f}s\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 14. TUNING DE HIPERPARÂMETROS — BÔNUS\n",
                "\n",
                "Busca sistemática de hiperparâmetros realizada via **`TrainValidationSplit`** sobre o modelo `RandomForestClassifier`."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14.1 Modelo Escolhido\n",
                "\n",
                "**`MODELO_BASE_ESCOLHIDO = RandomForestClassifier`**\n",
                "\n",
                "**Justificativa da Escolha:**\n",
                "A Random Forest superou a Regressão Logística em capacidade discriminativa global (**AUC-ROC de 0,8813 vs 0,8726**) e no **Recall da classe de alta rotatividade (84,99% vs 83,27%)**, apresentando menor tempo de treinamento e alta capacidade de capturar não-linearidades."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14.2 ParamGrid (8 Combinações Conservadoras)\n",
                "\n",
                "Construção da grade de hiperparâmetros conservadora com exatamente $2 \\times 2 \\times 2 = 8$ combinações:\n",
                "- **`numTrees`**: `[20, 30]`;\n",
                "- **`maxDepth`**: `[5, 8]`;\n",
                "- **`minInstancesPerNode`**: `[1, 5]`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rf = RandomForestClassifier(featuresCol='features', labelCol=TARGET_COL, seed=SEED)\n",
                "pipeline_rf = Pipeline(stages=indexers + encoders + [assembler, rf])\n",
                "\n",
                "paramGrid = ParamGridBuilder() \\\n",
                "    .addGrid(rf.numTrees, [20, 30]) \\\n",
                "    .addGrid(rf.maxDepth, [5, 8]) \\\n",
                "    .addGrid(rf.minInstancesPerNode, [1, 5]) \\\n",
                "    .build()\n",
                "\n",
                "NUMERO_COMBINACOES = len(paramGrid)\n",
                "print(f\"Total de combinações geradas pelo ParamGridBuilder: {NUMERO_COMBINACOES}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14.3 TrainValidationSplit\n",
                "\n",
                "\"TrainValidationSplit foi escolhido por possuir menor custo computacional que CrossValidator, sendo adequado ao ambiente local com 8 GB de RAM.\"\n",
                "\n",
                "- Subdivisão interna de validação: `trainRatio = 0.7` (70% treino interno / 30% validação interna);\n",
                "- Métrica de Avaliação: `areaUnderROC`;\n",
                "- Execução sequencial: `parallelism = 1` para prevenção de estouro de memória no JVM/driver."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tvs = TrainValidationSplit(\n",
                "    estimator=pipeline_rf,\n",
                "    estimatorParamMaps=paramGrid,\n",
                "    evaluator=eval_auc,\n",
                "    trainRatio=0.7,\n",
                "    seed=SEED,\n",
                "    parallelism=1\n",
                ")\n",
                "print(\"TrainValidationSplit instanciado com sucesso.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14.4 Tempo de Execução e Treinamento\n",
                "\n",
                "Execução do tuning exclusivamente no conjunto de **Treino** (`train_df`). O conjunto de **Teste** (`test_df`) foi mantido estritamente isolado."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "start_tvs = time.perf_counter()\n",
                "tvs_model = tvs.fit(train_df)\n",
                "TEMPO_TUNING = time.perf_counter() - start_tvs\n",
                "\n",
                "FATOR_CUSTO = TEMPO_TUNING / TEMPO_TREINO_BASE\n",
                "print(f\"Tuning concluído com sucesso!\")\n",
                "print(f\"  - TEMPO_TUNING = {TEMPO_TUNING:.3f} segundos ({TEMPO_TUNING/60.0:.2f} minutos)\")\n",
                "print(f\"  - FATOR_CUSTO  = {FATOR_CUSTO:.2f}x o tempo de treinamento base\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14.5 Melhores Hiperparâmetros Encontrados e Métricas Internas\n",
                "\n",
                "Métricas de AUC de validação interna registradas para as 8 combinações testadas:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "val_metrics = tvs_model.validationMetrics\n",
                "print(\"=== MÉTRICAS INTERNAS DE VALIDAÇÃO DO TRAINVALIDATIONSPLIT ===\")\n",
                "for i, (p_map, val_auc) in enumerate(zip(paramGrid, val_metrics), 1):\n",
                "    t = p_map[rf.numTrees]\n",
                "    d = p_map[rf.maxDepth]\n",
                "    m = p_map[rf.minInstancesPerNode]\n",
                "    print(f\"  Combinação {i}: numTrees={t:2d}, maxDepth={d:2d}, minInstancesPerNode={m:2d} | AUC Validação = {val_auc:.6f}\")\n",
                "\n",
                "best_pipeline = tvs_model.bestModel\n",
                "best_rf = best_pipeline.stages[-1]\n",
                "\n",
                "MELHOR_NUM_TREES = best_rf.getNumTrees\n",
                "MELHOR_MAX_DEPTH = best_rf.getMaxDepth()\n",
                "MELHOR_MIN_INST = best_rf.getMinInstancesPerNode()\n",
                "\n",
                "print(f\"\\n=== MELHORES HIPERPARÂMETROS SELECIONADOS ===\")\n",
                "print(f\"  numTrees            = {MELHOR_NUM_TREES}\")\n",
                "print(f\"  maxDepth            = {MELHOR_MAX_DEPTH}\")\n",
                "print(f\"  minInstancesPerNode = {MELHOR_MIN_INST}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14.6 Avaliação Final no Conjunto de Teste\n",
                "\n",
                "Aplicação do `bestModel` sobre o conjunto de **Teste** (`test_df`) para avaliação não viesada."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "preds_tuned = best_pipeline.transform(test_df)\n",
                "\n",
                "AUC_ROC_TUNED = eval_auc.evaluate(preds_tuned)\n",
                "ACCURACY_TUNED = eval_acc.evaluate(preds_tuned)\n",
                "\n",
                "tn_t = preds_tuned.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 0.0)).count()\n",
                "fp_t = preds_tuned.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 1.0)).count()\n",
                "fn_t = preds_tuned.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 0.0)).count()\n",
                "tp_t = preds_tuned.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 1.0)).count()\n",
                "\n",
                "prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0\n",
                "rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0\n",
                "f1_t = (2 * prec_t * rec_t / (prec_t + rec_t)) if (prec_t + rec_t) > 0 else 0.0\n",
                "spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0\n",
                "\n",
                "print(\"=== MATRIZ DE CONFUSÃO DA RANDOM FOREST (TUNED) ===\")\n",
                "print(f\"  Verdadeiros Negativos (TN_TUNED) = {tn_t:6,}\")\n",
                "print(f\"  Falsos Positivos (FP_TUNED)      = {fp_t:6,}\")\n",
                "print(f\"  Falsos Negativos (FN_TUNED)      = {fn_t:6,}\")\n",
                "print(f\"  Verdadeiros Positivos (TP_TUNED) = {tp_t:6,}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14.7 Comparação Antes × Depois\n",
                "\n",
                "| Métrica | Antes (RF Base) | Depois (RF Tuned) | Diferença |\n",
                "|---|---|---|---|\n",
                "| **AUC-ROC** | 0.8813 | 0.8813 | 0.0000 |\n",
                "| **Accuracy** | 78.56% | 78.56% | +0.00% |\n",
                "| **Precision (Classe 1)** | 0.7577 | 0.7577 | 0.0000 |\n",
                "| **Recall (Classe 1)** | 0.8499 | 0.8499 | 0.0000 |\n",
                "| **F1-Score (Classe 1)** | 0.8012 | 0.8012 | 0.0000 |\n",
                "| **Especificidade** | 0.7192 | 0.7192 | 0.0000 |\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14.8 Custo Computacional\n",
                "\n",
                "- **Tempo de Treinamento Padrão (RF Base):** 15,965 segundos;\n",
                "- **Tempo de Execução do Tuning (8 combinações TVS):** 52,800 segundos (0,88 minutos);\n",
                "- **Fator de Custo:** **3,31x** o tempo de treinamento padrão."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14.9 O Tuning Valeu a Pena?\n",
                "\n",
                "**`TUNING_VALEU_A_PENA = RESULTADO MISTO`**\n",
                "\n",
                "**Análise Crítica:**\n",
                "1. **Validação Empírica da Escolha Inicial:** O procedimento de busca em grade via `TrainValidationSplit` confirmou empiricamente que a configuração inicial conservadora adotada na Seção 13 (`numTrees=30`, `maxDepth=8`, `minInstancesPerNode=1`) já correspondia exatamente ao ponto ótimo dentro da grade examinada;\n",
                "2. **Estabilidade Preditiva:** As métricas de AUC-ROC (0,8813) e Acurácia (78,56%) mantiveram-se perfeitamente consistentes, comprovando ausência de overfitting na parametrização base;\n",
                "3. **Análise de Custo-Benefício Computacional:** Embora a execução do tuning no PySpark tenha demandado um fator de custo de **3,31x** o tempo padrão, o experimento valeu a pena do ponto de vista metodológico por comprovar a otimalidade dos hiperparâmetros sem comprometer o consumo de memória em 8 GB RAM."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tuned_dir = root_dir / 'models' / 'tuned_best_model_nordeste'\n",
                "if tuned_dir.exists():\n",
                "    shutil.rmtree(tuned_dir)\n",
                "\n",
                "print(f\"Salvando modelo tunado em: {tuned_dir}\")\n",
                "best_pipeline.write().overwrite().save(str(tuned_dir))\n",
                "MODELO_TUNED_SALVO = 'SIM' if (tuned_dir / 'metadata').exists() else 'NÃO'\n",
                "\n",
                "metrics_csv = root_dir / 'outputs' / 'metrics' / 'model_comparison_metrics.csv'\n",
                "rows = []\n",
                "if metrics_csv.exists():\n",
                "    with open(metrics_csv, 'r', encoding='utf-8') as f:\n",
                "        rows = list(csv.reader(f))\n",
                "\n",
                "with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:\n",
                "    writer = csv.writer(f)\n",
                "    if not rows:\n",
                "        writer.writerow(['model', 'auc_roc', 'accuracy', 'precision_pos', 'recall_pos', 'f1_pos', 'specificity', 'training_time_sec', 'seed'])\n",
                "    else:\n",
                "        for r in rows:\n",
                "            if r and r[0] != 'Random Forest (Tuned)':\n",
                "                writer.writerow(r)\n",
                "    writer.writerow(['Random Forest (Tuned)', f\"{AUC_ROC_TUNED:.6f}\", f\"{ACCURACY_TUNED:.6f}\", f\"{prec_t:.6f}\", f\"{rec_t:.6f}\", f\"{f1_t:.6f}\", f\"{spec_t:.6f}\", f\"{TEMPO_TUNING:.3f}\", SEED])\n",
                "\n",
                "print(f\"MODELO_TUNED_SALVO = {MODELO_TUNED_SALVO}\")\n",
                "print(f\"METRICAS_ATUALIZADAS = SIM ({metrics_csv})\")\n",
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
