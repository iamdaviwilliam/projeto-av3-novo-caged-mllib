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
    print("=== MONTAGEM DO NOTEBOOK OFICIAL COM SEÇÕES 1 A 13 (RF NORDESTE) ===")

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
                "## 12. PIPELINE MLLIB — LOGISTIC REGRESSION\n",
                "\n",
                "Pipeline de pré-processamento e treinamento do primeiro modelo de referência (Regressão Logística).\n",
                "- **Treino:** 23.510 registros (70,25%) | **Teste:** 9.955 registros (29,75%)\n",
                "- **AUC-ROC LR:** 0.8726 | **Acurácia LR:** 78,90%\n",
                "- **Tempo de Treino:** 15,465 segundos"
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
                "lr = LogisticRegression(featuresCol='features', labelCol=TARGET_COL)\n",
                "pipeline_lr = Pipeline(stages=indexers + encoders + [assembler, lr])\n",
                "\n",
                "start_lr = time.perf_counter()\n",
                "model_lr = pipeline_lr.fit(train_df)\n",
                "TEMPO_TREINO_LR = time.perf_counter() - start_lr\n",
                "\n",
                "preds_lr = model_lr.transform(test_df)\n",
                "eval_auc = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol='rawPrediction', metricName='areaUnderROC')\n",
                "eval_acc = MulticlassClassificationEvaluator(labelCol=TARGET_COL, predictionCol='prediction', metricName='accuracy')\n",
                "\n",
                "AUC_ROC_LR = eval_auc.evaluate(preds_lr)\n",
                "ACCURACY_LR = eval_acc.evaluate(preds_lr)\n",
                "\n",
                "tn_lr = preds_lr.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 0.0)).count()\n",
                "fp_lr = preds_lr.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 1.0)).count()\n",
                "fn_lr = preds_lr.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 0.0)).count()\n",
                "tp_lr = preds_lr.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 1.0)).count()\n",
                "\n",
                "prec_lr = tp_lr / (tp_lr + fp_lr)\n",
                "rec_lr = tp_lr / (tp_lr + fn_lr)\n",
                "f1_lr = (2 * prec_lr * rec_lr) / (prec_lr + rec_lr)\n",
                "spec_lr = tn_lr / (tn_lr + fp_lr)\n",
                "\n",
                "print(f\"Logistic Regression re-executada: AUC-ROC = {AUC_ROC_LR:.4f} | Accuracy = {ACCURACY_LR:.4f} ({ACCURACY_LR*100:.2f}%)\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. SEGUNDO MODELO — RANDOM FOREST CLASSIFIER\n",
                "\n",
                "Treinamento, avaliação, extração de importâncias de atributos e comparação direta da **Random Forest Classifier** com a Regressão Logística e o Baseline."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.1 Configuração Conservadora do Modelo\n",
                "\n",
                "Instanciação do `RandomForestClassifier` com parâmetros conservadores ajustados à máquina local (8 GB RAM):\n",
                "- `numTrees` = `30`;\n",
                "- `maxDepth` = `8`;\n",
                "- `seed` = `42`;\n",
                "- `featuresCol` = `'features'`;\n",
                "- Reutilização estrita dos mesmos subconjuntos de **Treino** (23.510 registros) e **Teste** (9.955 registros).\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rf = RandomForestClassifier(featuresCol='features', labelCol=TARGET_COL, numTrees=30, maxDepth=8, seed=SEED)\n",
                "pipeline_rf = Pipeline(stages=indexers + encoders + [assembler, rf])\n",
                "print(\"Pipeline PySpark MLlib para Random Forest montado com sucesso.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.2 Treinamento da Random Forest\n",
                "\n",
                "Ajuste do ensemble de árvores de decisão exclusivamente no conjunto de **Treino** (`train_df`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "start_rf = time.perf_counter()\n",
                "model_rf = pipeline_rf.fit(train_df)\n",
                "TEMPO_TREINO_RF = time.perf_counter() - start_rf\n",
                "\n",
                "rf_stage = model_rf.stages[-1]\n",
                "print(f\"Treinamento da Random Forest concluído em TEMPO_TREINO_RF = {TEMPO_TREINO_RF:.3f} segundos!\")\n",
                "print(f\"Número de Árvores: {rf_stage.getNumTrees} | Profundidade Máxima: {rf_stage.getMaxDepth()}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.3 Avaliação no Conjunto de Teste\n",
                "\n",
                "Aplicação do modelo `model_rf` no conjunto de **Teste** (`test_df`) para cálculo de AUC-ROC e Acurácia."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "predictions_rf = model_rf.transform(test_df)\n",
                "\n",
                "AUC_ROC_RF = eval_auc.evaluate(predictions_rf)\n",
                "ACCURACY_RF = eval_acc.evaluate(predictions_rf)\n",
                "\n",
                "print(f\"=== DESEMPENHO DA RANDOM FOREST NO CONJUNTO DE TESTE ===\")\n",
                "print(f\"  AUC-ROC_RF  = {AUC_ROC_RF:.4f}\")\n",
                "print(f\"  ACCURACY_RF = {ACCURACY_RF:.4f} ({ACCURACY_RF*100:.2f}%)\")\n",
                "\n",
                "print(\"\\nAmostra de Predições da Random Forest (5 registros):\")\n",
                "predictions_rf.select(TARGET_COL, 'prediction', 'probability').show(5, truncate=False)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.4 Matriz de Confusão e Métricas Detalhadas\n",
                "\n",
                "Cálculo dos acertos e erros na Matriz de Confusão para o modelo Random Forest."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tn_rf = predictions_rf.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 0.0)).count()\n",
                "fp_rf = predictions_rf.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 1.0)).count()\n",
                "fn_rf = predictions_rf.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 0.0)).count()\n",
                "tp_rf = predictions_rf.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 1.0)).count()\n",
                "\n",
                "precision_rf = tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0.0\n",
                "recall_rf = tp_rf / (tp_rf + fn_rf) if (tp_rf + fn_rf) > 0 else 0.0\n",
                "f1_rf = (2 * precision_rf * recall_rf / (precision_rf + recall_rf)) if (precision_rf + recall_rf) > 0 else 0.0\n",
                "specificity_rf = tn_rf / (tn_rf + fp_rf) if (tn_rf + fp_rf) > 0 else 0.0\n",
                "\n",
                "print(\"=== MATRIZ DE CONFUSÃO DA RANDOM FOREST ===\")\n",
                "print(f\"  Verdadeiros Negativos (TN_RF) = {tn_rf:6,}\")\n",
                "print(f\"  Falsos Positivos (FP_RF)      = {fp_rf:6,}\")\n",
                "print(f\"  Falsos Negativos (FN_RF)      = {fn_rf:6,}\")\n",
                "print(f\"  Verdadeiros Positivos (TP_RF) = {tp_rf:6,}\")\n",
                "\n",
                "print(\"\\n=== MÉTRICAS DA CLASSE POSITIVA (RANDOM FOREST) ===\")\n",
                "print(f\"  Precision Classe 1 = {precision_rf:.4f}\")\n",
                "print(f\"  Recall Classe 1    = {recall_rf:.4f}\")\n",
                "print(f\"  F1-Score Classe 1  = {f1_rf:.4f}\")\n",
                "print(f\"  Especificidade     = {specificity_rf:.4f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.5 Comparação LR × RF × Baseline\n",
                "\n",
                "| Métrica | Baseline (Classe Majoritária 0) | Logistic Regression | Random Forest (Ensemble) |\n",
                "|---|---|---|---|\n",
                "| **AUC-ROC** | N/A | 0.8726 | **0.8813** (+$+0,87\\%$) |\n",
                "| **Accuracy** | 49.19% | **78.90%** | 78.56% |\n",
                "| **Precision (Classe 1)** | 0.0000 | **0.7704** | 0.7577 |\n",
                "| **Recall (Classe 1)** | 0.0000 | 0.8327 | **0.8499** (+$+1,72\\%$) |\n",
                "| **F1-Score (Classe 1)** | 0.0000 | 0.8004 | **0.8012** (+$+0,08\\%$) |\n",
                "| **Especificidade** | 1.0000 | **0.7437** | 0.7192 |\n",
                "| **Tempo de Treino** | N/A | 15.465s | **9.605s** ($-37,9\\%$) |\n",
                "\n",
                "**Análise Comparativa:**\n",
                "- A **Random Forest** superou a Regressão Logística em discriminação global (**AUC-ROC de 0,8813 vs 0,8726**), **Recall da classe 1 (84,99% vs 83,27%)** e **F1-Score (0,8012 vs 0,8004)**.\n",
                "- A **Regressão Logística** manteve ligeira vantagem em **Acurácia (78,90% vs 78,56%)** e **Especificidade (74,37% vs 71,92%)**.\n",
                "- No que tange ao tempo de execução no PySpark, a Random Forest convergiu mais rapidamente (**9,605s vs 15,465s** da LR)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.6 Feature Importance (Random Forest)\n",
                "\n",
                "Mapeamento exato entre os índices do vetor `features` e os nomes dos atributos One-Hot e numéricos.\n",
                "\n",
                "> [!IMPORTANT]\n",
                "> **Aviso Metodológico:** A importância preditiva gerada pelas árvores de decisão mede a capacidade de divisão de variância dos nós e **não representa efeito causal**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "field_meta = predictions_rf.schema['features'].metadata.get('ml_attr', {}).get('attrs', {})\n",
                "idx_to_name = {}\n",
                "for attr_type in ['numeric', 'binary']:\n",
                "    if attr_type in field_meta:\n",
                "        for item in field_meta[attr_type]:\n",
                "            idx_to_name[item['idx']] = item['name']\n",
                "\n",
                "feature_names = [idx_to_name.get(i, f\"feature_{i}\") for i in range(len(idx_to_name))]\n",
                "importances = rf_stage.featureImportances.toArray()\n",
                "rf_feat_imp = list(zip(range(len(importances)), feature_names, importances))\n",
                "rf_feat_imp_sorted = sorted(rf_feat_imp, key=lambda x: x[2], reverse=True)\n",
                "\n",
                "print(\"=== TOP 10 FEATURES MAIS IMPORTANTES (RANDOM FOREST) ===\")\n",
                "for rank, (idx, fname, imp) in enumerate(rf_feat_imp_sorted[:10], 1):\n",
                "    print(f\"  {rank:2d}. {fname:35s} | Importância: {imp:.6f}\")\n",
                "\n",
                "print(\"\\n=== IMPORTÂNCIA AGREGADA POR FEATURE ORIGINAL ===\")\n",
                "agg_imp = {}\n",
                "for idx, fname, imp in rf_feat_imp:\n",
                "    orig = fname.split('_')[0]\n",
                "    if 'FAIXA_ETARIA' in fname: orig = 'FAIXA_ETARIA'\n",
                "    elif 'FAIXA_VOLUME' in fname: orig = 'FAIXA_VOLUME_COORTE'\n",
                "    elif 'LOG_VOLUME' in fname: orig = 'LOG_VOLUME_COORTE'\n",
                "    elif 'PROP_NEGATIVOS' in fname: orig = 'PROP_NEGATIVOS_T'\n",
                "    elif 'MES_SIN' in fname: orig = 'MES_SIN'\n",
                "    elif 'MES_COS' in fname: orig = 'MES_COS'\n",
                "    elif 'N_TOTAL' in fname: orig = 'N_TOTAL_T'\n",
                "    elif 'N_POSITIVOS' in fname: orig = 'N_POSITIVOS_T'\n",
                "    elif 'N_NEGATIVOS' in fname: orig = 'N_NEGATIVOS_T'\n",
                "    agg_imp[orig] = agg_imp.get(orig, 0.0) + imp\n",
                "\n",
                "for orig_f, imp_sum in sorted(agg_imp.items(), key=lambda x: x[1], reverse=True):\n",
                "    print(f\"  {orig_f:25s} | Importância Agregada: {imp_sum:.6f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.7 Top 5 Features da Random Forest\n",
                "\n",
                "| Ranking | Feature | Importância | Interpretação no Mercado Formal do Nordeste |\n",
                "|---|---|---|---|\n",
                "| 1 | `PROP_NEGATIVOS_T` | 0.194124 | A proporção instantânea de desligamentos no mês de referência $t_0$ reflete a inércia imediata de rotatividade do setor |\n",
                "| 2 | `FAIXA_ETARIA_18-24` | 0.171897 | Trabalhadores jovens de primeiro emprego apresentam maior mobilidade e volatilidade contratual |\n",
                "| 3 | `FAIXA_ETARIA_65+` | 0.171618 | Faixa etária sênior associada a aposentadorias, desligamentos definitivos e saídas do mercado formal |\n",
                "| 4 | `FAIXA_ETARIA_14-17` | 0.132897 | Aprendizes e contratos de menor duração com encerramentos previstos por prazo determinado |\n",
                "| 5 | `FAIXA_ETARIA_55-64` | 0.091816 | Trabalhadores pré-aposentadoria com dinâmicas específicas de transição de carreira no Nordeste |\n",
                "\n",
                "*Importância preditiva não deve ser interpretada como efeito causal.*"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.8 Coeficientes da Logistic Regression (Top 5)\n",
                "\n",
                "Mapeamento e ordenação dos coeficientes em magnitude absoluta do primeiro modelo (Regressão Logística).\n",
                "\n",
                "| Ranking | Feature | Coeficiente | Sinal | Interpretação |\n",
                "|---|---|---|---|---|\n",
                "| 1 | `FAIXA_ETARIA_14-17` | -2.997699 | Negativo (-) | Associação com menor log-odds de rotatividade sustentada por 6 meses consecutivos |\n",
                "| 2 | `FAIXA_ETARIA_65+` | +2.686121 | Positivo (+) | Forte associação com maior probabilidade de desligamentos acumulados em 6 meses |\n",
                "| 3 | `FAIXA_ETARIA_18-24` | -2.615496 | Negativo (-) | Associação negativa na persistência contínua dos 6 meses subsequentes |\n",
                "| 4 | `seção_G` (Comércio) | +1.708004 | Positivo (+) | Setor de comércio e reparação automotiva no Nordeste possui alta rotatividade contratual |\n",
                "| 5 | `FAIXA_ETARIA_55-64` | +1.570645 | Positivo (+) | Aumento nos log-odds de desligamentos futuros |\n",
                "\n",
                "*Nota sobre Escalas:* Como as features numéricas possuem escalas variadas, a magnitude dos coeficientes entre variáveis contínuas e variáveis One-Hot dummy deve ser analisada com cautela."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "lr_stage = model_lr.stages[-1]\n",
                "coeffs = lr_stage.coefficients.toArray()\n",
                "lr_feat_coef = list(zip(range(len(coeffs)), feature_names, coeffs, [abs(c) for c in coeffs]))\n",
                "lr_feat_coef_sorted = sorted(lr_feat_coef, key=lambda x: x[3], reverse=True)\n",
                "\n",
                "print(\"=== TOP 10 COEFICIENTES EM MAGNITUDE ABSOLUTA (LOGISTIC REGRESSION) ===\")\n",
                "for rank, (idx, fname, coef, abs_c) in enumerate(lr_feat_coef_sorted[:10], 1):\n",
                "    sinal = '+' if coef > 0 else '-'\n",
                "    print(f\"  {rank:2d}. {fname:35s} | Coeficiente: {coef:10.6f} | Sinal: {sinal}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.9 Modelo Recomendado\n",
                "\n",
                "**`MODELO_RECOMENDADO = RandomForestClassifier`**\n",
                "\n",
                "**Justificativa Técnica:**\n",
                "1. **Maior Capacidade Discriminativa:** A Random Forest alcançou **AUC-ROC de 0,8813** no conjunto de teste (superando 0,8726 da Regressão Logística);\n",
                "2. **Superioridade no Recall da Classe Positiva:** Atingiu **84,99% de Recall** na detecção de coortes de alta rotatividade futura (vs 83,27% da LR), minimizando falsos negativos na gestão pública e corporativa;\n",
                "3. **Eficiência Computacional no PySpark:** O treinamento convergiu em apenas **9,605 segundos** (contra 15,465 segundos da LR);\n",
                "4. **Robustez Não Linear:** Captura naturalmente as interações não lineares entre as faixas etárias e os setores econômicos da Região Nordeste."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13.10 Limitações e Persistência da Random Forest\n",
                "\n",
                "**Limitações:**\n",
                "- O modelo foi avaliado com hyperparâmetros conservadores (`numTrees=30`, `maxDepth=8`) sem busca em grade ou validação cruzada para preservar estabilidade de memória em 8 GB RAM.\n",
                "\n",
                "**Persistência do Modelo e Métricas Atualizadas:**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rf_model_dir = root_dir / 'models' / 'random_forest_nordeste'\n",
                "if rf_model_dir.exists():\n",
                "    shutil.rmtree(rf_model_dir)\n",
                "\n",
                "print(f\"Salvando PipelineModel da Random Forest em: {rf_model_dir}\")\n",
                "model_rf.write().overwrite().save(str(rf_model_dir))\n",
                "MODELO_RF_SALVO = 'SIM' if (rf_model_dir / 'metadata').exists() else 'NÃO'\n",
                "\n",
                "metrics_csv = root_dir / 'outputs' / 'metrics' / 'model_comparison_metrics.csv'\n",
                "metrics_csv.parent.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:\n",
                "    writer = csv.writer(f)\n",
                "    writer.writerow(['model', 'auc_roc', 'accuracy', 'precision_pos', 'recall_pos', 'f1_pos', 'specificity', 'training_time_sec', 'seed'])\n",
                "    writer.writerow(['Baseline (Majoritária)', f\"0.000000\", f\"{0.491914:.6f}\", f\"0.000000\", f\"0.000000\", f\"0.000000\", f\"1.000000\", f\"0.000\", SEED])\n",
                "    writer.writerow(['Logistic Regression', f\"{AUC_ROC_LR:.6f}\", f\"{ACCURACY_LR:.6f}\", f\"{prec_lr:.6f}\", f\"{rec_lr:.6f}\", f\"{f1_lr:.6f}\", f\"{spec_lr:.6f}\", f\"{TEMPO_TREINO_LR:.3f}\", SEED])\n",
                "    writer.writerow(['Random Forest', f\"{AUC_ROC_RF:.6f}\", f\"{ACCURACY_RF:.6f}\", f\"{precision_rf:.6f}\", f\"{recall_rf:.6f}\", f\"{f1_rf:.6f}\", f\"{specificity_rf:.6f}\", f\"{TEMPO_TREINO_RF:.3f}\", SEED])\n",
                "\n",
                "METRICAS_ATUALIZADAS = 'SIM' if metrics_csv.exists() else 'NÃO'\n",
                "print(f\"MODELO_RF_SALVO = {MODELO_RF_SALVO}\")\n",
                "print(f\"METRICAS_ATUALIZADAS = {METRICAS_ATUALIZADAS} ({metrics_csv})\")\n",
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
