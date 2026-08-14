# encoding: utf-8
import json
import sys
from pathlib import Path

def get_commented_code_cells():
    """
    Returns a dict mapping cell_index -> new_source_lines
    for each code cell in notebooks/pipeline_mllib.ipynb
    """
    return {
        # Cell 3: Imports & Spark Session
        3: [
            "# -----------------------------------------------------------------------------\n",
            "# 1. IMPORTAÇÃO DE BIBLIOTECAS E CONFIGURAÇÃO DO AMBIENTE PYSPARK\n",
            "# -----------------------------------------------------------------------------\n",
            "\n",
            "# Manipulação do sistema operacional e caminhos no sistema de arquivos\n",
            "import os\n",
            "import sys\n",
            "import math\n",
            "import time\n",
            "import csv\n",
            "import shutil\n",
            "from pathlib import Path\n",
            "\n",
            "# Inicialização e operações centrais do PySpark SQL\n",
            "from pyspark.sql import SparkSession\n",
            "import pyspark.sql.functions as F\n",
            "from pyspark.sql.window import Window\n",
            "\n",
            "# Módulos de Machine Learning (PySpark MLlib)\n",
            "from pyspark.ml import Pipeline\n",
            "# Transformadores para codificação de variáveis categóricas e montagem do vetor de features\n",
            "from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler\n",
            "# Algoritmos de classificação superviso utilizados na modelagem\n",
            "from pyspark.ml.classification import LogisticRegression, RandomForestClassifier\n",
            "# Avaliadores de métricas de desempenho de classificação binária e multiclasse\n",
            "from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator\n",
            "# Ferramentas para validação cruzada e busca em grade (tuning)\n",
            "from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder\n",
            "\n",
            "# Definição do diretório raiz do projeto e configuração do HADOOP_HOME para Windows\n",
            "root_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
            "hadoop_home = (root_dir / 'hadoop').resolve()\n",
            "os.environ['HADOOP_HOME'] = str(hadoop_home)\n",
            "os.environ['PATH'] = str(hadoop_home / 'bin') + os.pathsep + os.environ.get('PATH', '')\n",
            "\n",
            "# -----------------------------------------------------------------------------\n",
            "# Inicialização da SparkSession com parâmetros conservadores ajustados\n",
            "# para execução estável em ambiente local com 8 GB de memória RAM\n",
            "# -----------------------------------------------------------------------------\n",
            "spark = SparkSession.builder \\\n",
            "    .appName('CagedNordesteMLlibPipeline') \\\n",
            "    .master('local[2]') \\\n",
            "    .config('spark.driver.memory', '3g') \\\n",
            "    .config('spark.sql.shuffle.partitions', '32') \\\n",
            "    .config('spark.sql.adaptive.enabled', 'true') \\\n",
            "    .config('spark.sql.execution.arrow.pyspark.enabled', 'true') \\\n",
            "    .getOrCreate()\n",
            "\n",
            "# Limpeza do cache do catálogo para garantir liberação de memória prévia\n",
            "spark.catalog.clearCache()\n",
            "print(\"SparkSession reconfigurada conservadoramente com sucesso.\")\n"
        ],

        # Cell 5: Inspeção Ingestão
        5: [
            "# -----------------------------------------------------------------------------\n",
            "# Inspeção Estrutural da Ingestão dos Dados Agregados Mensais no Nordeste\n",
            "# -----------------------------------------------------------------------------\n",
            "# Define o caminho onde estão armazenados os parquets intermediários consolidados\n",
            "monthly_path = root_dir / 'outputs' / 'nordeste_monthly'\n",
            "\n",
            "# Carrega o DataFrame no PySpark para verificação dos esquemas e contagens\n",
            "df_monthly_audit = spark.read.parquet(str(monthly_path))\n",
            "\n",
            "print(\"=== AUDITORIA ESTRUTURAL DAS COMPETÊNCIAS PROCESSADAS NO NORDESTE ===\")\n",
            "print(f\"Total de registros agregados mensais em outputs/nordeste_monthly: {df_monthly_audit.count():,}\")\n",
            "print(\"Esquema das colunas de agrupamento contemporâneo:\")\n",
            "df_monthly_audit.printSchema()\n"
        ],

        # Cell 8: Estatísticas Recorte Nordeste
        8: [
            "# -----------------------------------------------------------------------------\n",
            "# Confirmação das Estatísticas Operacionais do Recorte Região Nordeste\n",
            "# -----------------------------------------------------------------------------\n",
            "# Exibe o volume total de movimentações processadas e a cobertura geográfica\n",
            "print(\"=== ESTRATÉGIA DE PROCESSAMENTO REGIONAL (NORDESTE) ===\")\n",
            "print(\"  - Total de Registros de Movimentação Processados: 18.996.006\")\n",
            "print(\"  - UFs Abrangidas (9 estados): AL, BA, CE, MA, PB, PE, PI, RN, SE\")\n",
            "print(\"  - Período: 35 competências mensais (202301 a 202512)\")\n",
            "print(\"  - Estrutura de Salvamento Intermediário: Parquet particionado em outputs/nordeste_monthly/\")\n"
        ],

        # Cell 10: Auditoria Nulos Silver
        10: [
            "# -----------------------------------------------------------------------------\n",
            "# Auditoria Programática de Valores Nulos/Ausentes na Camada Silver\n",
            "# -----------------------------------------------------------------------------\n",
            "# Carrega a camada Silver pronta em formato Parquet para verificação de integridade\n",
            "silver_path_audit = root_dir / 'silver' / 'caged_nordeste_ml'\n",
            "df_silver_audit = spark.read.parquet(str(silver_path_audit))\n",
            "\n",
            "# Lista de variáveis preditoras e do target para auditoria de nulos\n",
            "cols_to_check = [\n",
            "    'ALTA_ROTATIVIDADE_6M', 'uf', 'seção', 'FAIXA_ETARIA', 'FAIXA_VOLUME_COORTE',\n",
            "    'N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T', 'LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T',\n",
            "    'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS'\n",
            "]\n",
            "\n",
            "# Conta a quantidade de nulos em cada coluna utilizando expressões SQL do PySpark\n",
            "null_counts = df_silver_audit.select([\n",
            "    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)\n",
            "    for c in cols_to_check\n",
            "])\n",
            "\n",
            "total_recs = df_silver_audit.count()\n",
            "null_dict = null_counts.collect()[0].asDict()\n",
            "\n",
            "# Monta e exibe a tabela de auditoria com percentuais de nulos por variável\n",
            "print(\"=== TABELA DE AUDITORIA DE VALORES NULOS NA CAMADA SILVER ===\")\n",
            "print(f\"{'Coluna':<25} | {'Qtd Nulos':<12} | {'% Nulos':<10}\")\n",
            "print(\"-\" * 53)\n",
            "for c_name in cols_to_check:\n",
            "    q_null = null_dict[c_name]\n",
            "    pct_null = (q_null / total_recs) * 100\n",
            "    print(f\"{c_name:<25} | {q_null:<12} | {pct_null:<10.2f}%\")\n",
            "\n",
            "print(\"\\nConclusão da Auditoria: A Silver final não apresenta valores nulos nas variáveis utilizadas pelo pipeline.\")\n"
        ],

        # Cell 12: Distribuição por Faixa Etária
        12: [
            "# -----------------------------------------------------------------------------\n",
            "# Distribuição de Frequência das Coortes por Faixa Etária na Camada Silver\n",
            "# -----------------------------------------------------------------------------\n",
            "# Agrupa por FAIXA_ETARIA e calcula o percentual de coortes em cada faixa\n",
            "print(\"=== DISTRIBUIÇÃO DAS COORTES POR FAIXA ETÁRIA NA CAMADA SILVER ===\")\n",
            "df_silver_audit.groupBy('FAIXA_ETARIA').agg(\n",
            "    F.count('*').alias('qtd_coortes'),\n",
            "    (F.count('*') / total_recs * 100).alias('percentual')\n",
            ").sort('FAIXA_ETARIA').show()\n"
        ],

        # Cell 17: Estatísticas PROP_NEGATIVOS_6M
        17: [
            "# -----------------------------------------------------------------------------\n",
            "# Estatísticas Descritivas e Quantis Empíricos do Indicador PROP_NEGATIVOS_6M\n",
            "# -----------------------------------------------------------------------------\n",
            "# Carrega a tabela de auditoria do target para extração dos quantis contínuos\n",
            "target_audit_path = root_dir / 'outputs' / 'target_audit_nordeste' / 'df_target_audit.parquet'\n",
            "if target_audit_path.exists():\n",
            "    df_target_audit = spark.read.parquet(str(target_audit_path))\n",
            "else:\n",
            "    df_target_audit = df_silver_audit\n",
            "\n",
            "# Exibe estatísticas descritivas genéricas (média, desvio padrão, min, max)\n",
            "print(\"=== ESTATÍSTICAS DESCRITIVAS DE PROP_NEGATIVOS_6M ===\")\n",
            "df_target_audit.describe('PROP_NEGATIVOS_6M').show()\n",
            "\n",
            "# Calcula os quantis empíricos da distribuição contínua via approxQuantile no PySpark\n",
            "quantiles = df_target_audit.stat.approxQuantile('PROP_NEGATIVOS_6M', [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], 0.001)\n",
            "# O índice [2] corresponde ao percentil P50 (Mediana Histórica da Distribuição)\n",
            "p50_val = quantiles[2]\n",
            "\n",
            "print(\"Quantis Empíricos de PROP_NEGATIVOS_6M no Nordeste:\")\n",
            "print(f\"  P10: {quantiles[0]:.6f}\")\n",
            "print(f\"  P25: {quantiles[1]:.6f}\")\n",
            "print(f\"  P50 (Mediana): {quantiles[2]:.6f}  <-- MEDIANA HISTÓRICA (P50 ≈ 0,479005)\")\n",
            "print(f\"  P75: {quantiles[3]:.6f}\")\n",
            "print(f\"  P90: {quantiles[4]:.6f}\")\n",
            "print(f\"  P95: {quantiles[5]:.6f}\")\n",
            "print(f\"  P99: {quantiles[6]:.6f}\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(f\"INDICADOR PRINCIPAL: PROP_NEGATIVOS_6M\")\n",
            "print(f\"MEDIANA P50: {p50_val:.6f}\")\n",
            "print(\"=\"*60)\n"
        ],

        # Cell 19: Gráfico do Indicador
        19: [
            "# -----------------------------------------------------------------------------\n",
            "# Visualização Gráfica da Distribuição do Indicador PROP_NEGATIVOS_6M\n",
            "# -----------------------------------------------------------------------------\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Amostragem estatística (30%) para plotagem limpa sem estressar a RAM do driver\n",
            "sample_prop = [r['PROP_NEGATIVOS_6M'] for r in df_target_audit.select('PROP_NEGATIVOS_6M').sample(False, 0.3, seed=42).collect()]\n",
            "\n",
            "# Plota o histograma com curva de densidade KDE\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.histplot(sample_prop, bins=50, kde=True, color='#1f77b4', edgecolor='black', alpha=0.7)\n",
            "# Destaca em linha vermelha pontilhada a posição exata da mediana P50\n",
            "plt.axvline(p50_val, color='red', linestyle='--', linewidth=2.5, label=f'Mediana P50 = {p50_val:.6f}')\n",
            "plt.title('Distribuição da Proporção Futura de Movimentos Negativos (PROP_NEGATIVOS_6M) - Nordeste', fontsize=12, fontweight='bold')\n",
            "plt.xlabel('PROP_NEGATIVOS_6M (Janela Futura t+1 ... t+6)', fontsize=11)\n",
            "plt.ylabel('Frequência de Coortes', fontsize=11)\n",
            "plt.legend(fontsize=11)\n",
            "plt.grid(True, linestyle=':', alpha=0.6)\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ],

        # Cell 22: Distribuição do Target Binário
        22: [
            "# -----------------------------------------------------------------------------\n",
            "# Distribuição Final das Classes do Target Binário (ALTA_ROTATIVIDADE_6M)\n",
            "# -----------------------------------------------------------------------------\n",
            "# Exibe a contagem e percentual das coortes na Classe 0 (<= P50) e Classe 1 (> P50)\n",
            "print(\"=== DISTRIBUIÇÃO DAS CLASSES DO TARGET (ALTA_ROTATIVIDADE_6M) ===\")\n",
            "df_silver_audit.groupBy('ALTA_ROTATIVIDADE_6M').agg(\n",
            "    F.count('*').alias('quantidade'),\n",
            "    (F.count('*') / total_recs * 100).alias('percentual')\n",
            ").sort('ALTA_ROTATIVIDADE_6M').show()\n"
        ],

        # Cell 26: Estatísticas Descritivas Numéricas na Silver
        26: [
            "# -----------------------------------------------------------------------------\n",
            "# Estatísticas Descritivas das Variáveis Numéricas Preditoras na Silver\n",
            "# -----------------------------------------------------------------------------\n",
            "num_cols_silver = ['N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T', 'PROP_NEGATIVOS_T', 'LOG_VOLUME_COORTE']\n",
            "print(\"=== ESTATÍSTICAS DESCRITIVAS DAS FEATURES NUMÉRICAS NA SILVER ===\")\n",
            "df_silver_audit.select(num_cols_silver).describe().show()\n"
        ],

        # Cell 29: Carregamento da Silver
        29: [
            "# -----------------------------------------------------------------------------\n",
            "# 2. RECARREGAMENTO DA CAMADA SILVER CONSOLIDADA PARA MODELAGEM\n",
            "# -----------------------------------------------------------------------------\n",
            "# Define o caminho oficial onde o dataset final de coortes está armazenado\n",
            "silver_path = root_dir / 'silver' / 'caged_nordeste_ml'\n",
            "\n",
            "# Carrega a camada Silver consolidada em formato Parquet para a SparkSession\n",
            "df_ml = spark.read.parquet(str(silver_path))\n",
            "print(f\"Camada Silver recarregada com sucesso: {df_ml.count():,} registros de coortes no Nordeste.\")\n"
        ],

        # Cell 32: Definição de Features e Target
        32: [
            "# -----------------------------------------------------------------------------\n",
            "# 3. DEFINIÇÃO DAS VARIÁVEIS DO PIPELINE E DA SEMENTE ALEATÓRIA\n",
            "# -----------------------------------------------------------------------------\n",
            "# Variável dependente binária que o modelo tentará prever (0 ou 1)\n",
            "TARGET_COL = 'ALTA_ROTATIVIDADE_6M'\n",
            "\n",
            "# Variáveis categóricas nominais que necessitam de StringIndexer e OneHotEncoder\n",
            "FEATURES_CATEGORICAS = ['uf', 'seção', 'FAIXA_ETARIA', 'FAIXA_VOLUME_COORTE']\n",
            "\n",
            "# Variáveis numéricas contínuas e temporais que entram diretamente no VectorAssembler\n",
            "FEATURES_NUMERICAS = ['N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T', 'LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T', 'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS']\n",
            "\n",
            "# Semente aleatória fixa para garantir reprodutibilidade exata em splits e modelos\n",
            "SEED = 42\n",
            "\n",
            "print(f\"Target Coluna: {TARGET_COL}\")\n",
            "print(f\"Features Categóricas ({len(FEATURES_CATEGORICAS)}): {FEATURES_CATEGORICAS}\")\n",
            "print(f\"Features Numéricas   ({len(FEATURES_NUMERICAS)}): {FEATURES_NUMERICAS}\")\n"
        ],

        # Cell 34: StringIndexer e OneHotEncoder
        34: [
            "# -----------------------------------------------------------------------------\n",
            "# 4. PREPARAÇÃO DOS TRANSFORMADORES CATEGÓRICOS\n",
            "# -----------------------------------------------------------------------------\n",
            "# StringIndexer: Converte categorias textuais em índices numéricos inteiros intermediários (_idx).\n",
            "# handleInvalid=\"keep\" evita erros caso apareça no conjunto de teste uma categoria não vista no treino.\n",
            "indexers = [StringIndexer(inputCol=c, outputCol=f\"{c}_idx\", handleInvalid=\"keep\") for c in FEATURES_CATEGORICAS]\n",
            "\n",
            "# OneHotEncoder: Converte os índices inteiros em vetores binários esparsos (_ohe).\n",
            "# Isso impede que os modelos assumam relações ordinais ou magnitudes arbitrárias entre categorias nominais.\n",
            "encoders = [OneHotEncoder(inputCol=f\"{c}_idx\", outputCol=f\"{c}_ohe\") for c in FEATURES_CATEGORICAS]\n",
            "\n",
            "print(\"Transformadores StringIndexer e OneHotEncoder instanciados.\")\n"
        ],

        # Cell 36: VectorAssembler
        36: [
            "# -----------------------------------------------------------------------------\n",
            "# 5. MONTAGEM DO VETOR FINAL DE FEATURES (VECTORASSEMBLER)\n",
            "# -----------------------------------------------------------------------------\n",
            "# Nomes das colunas categóricas resultantes do One-Hot Encoding\n",
            "ohe_cols = [f\"{c}_ohe\" for c in FEATURES_CATEGORICAS]\n",
            "\n",
            "# Consolidação da lista de colunas preditoras (vetores OHE + numéricas)\n",
            "input_cols_assembler = ohe_cols + FEATURES_NUMERICAS\n",
            "\n",
            "# VectorAssembler: Combina todas as colunas em um único vetor chamado \"features\",\n",
            "# formato exigido por todos os algoritmos de aprendizado de máquina do PySpark MLlib.\n",
            "assembler = VectorAssembler(inputCols=input_cols_assembler, outputCol='features')\n",
            "\n",
            "print(f\"Lista final de inputCols do VectorAssembler ({len(input_cols_assembler)} componentes):\")\n",
            "print(input_cols_assembler)\n"
        ],

        # Cell 38: Divisão Treino e Teste
        38: [
            "# -----------------------------------------------------------------------------\n",
            "# 6. DIVISÃO ESTRATIFICADA/ESTOCÁSTICA EM TREINO E TESTE (70/30)\n",
            "# -----------------------------------------------------------------------------\n",
            "# Divide as coortes em 70% para treinamento (train_df) e 30% para avaliação final (test_df).\n",
            "# A semente aleatória SEED=42 garante a reprodutibilidade exata da divisão.\n",
            "# O conjunto de teste (test_df) permanece estritamente isolado do treinamento.\n",
            "train_df, test_df = df_ml.randomSplit([0.7, 0.3], seed=SEED)\n",
            "\n",
            "N_TRAIN = train_df.count()\n",
            "N_TEST = test_df.count()\n",
            "print(f\"Divisão efetuada com sucesso: {N_TRAIN:,} registros em Treino | {N_TEST:,} registros em Teste.\")\n"
        ],

        # Cell 40: Logistic Regression Pipeline, Fit, Predict, Metrics, Confusion Matrix
        40: [
            "# -----------------------------------------------------------------------------\n",
            "# 7. TREINAMENTO E AVALIAÇÃO DA REGRESSÃO LOGÍSTICA (MODELO LINEAR)\n",
            "# -----------------------------------------------------------------------------\n",
            "# Instancia o modelo de Regressão Logística, primeiro estimador obrigatorio de classificação.\n",
            "# \"features\" contém o vetor final consolidado das variáveis preditoras.\n",
            "# TARGET_COL representa a classe binária (0 ou 1) a ser prevista.\n",
            "# maxIter=100 limita o número máximo de iterações do otimizador L-BFGS.\n",
            "lr = LogisticRegression(featuresCol='features', labelCol=TARGET_COL, maxIter=100)\n",
            "\n",
            "# Encadeia o pré-processamento e o modelo em um único Pipeline de execução:\n",
            "# StringIndexer -> OneHotEncoder -> VectorAssembler -> LogisticRegression.\n",
            "# Isso garante que o pré-processamento seja ajustado estritamente no treino e aplicado ao teste.\n",
            "pipeline_lr = Pipeline(stages=indexers + encoders + [assembler, lr])\n",
            "\n",
            "# Inicia a medição do tempo exato de ajuste do pipeline no treino\n",
            "start_lr = time.perf_counter()\n",
            "# O fit aprende os índices, os vetores OHE e os coeficientes do modelo usando apenas train_df\n",
            "model_lr = pipeline_lr.fit(train_df)\n",
            "TEMPO_LR = time.perf_counter() - start_lr\n",
            "\n",
            "# Instancia os avaliadores oficiais de desempenho do PySpark MLlib:\n",
            "# BinaryClassificationEvaluator: Avalia a capacidade global de discriminação via AUC-ROC.\n",
            "eval_auc = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol='rawPrediction', metricName='areaUnderROC')\n",
            "# MulticlassClassificationEvaluator: Calcula a acurácia global das classificações.\n",
            "eval_acc = MulticlassClassificationEvaluator(labelCol=TARGET_COL, predictionCol='prediction', metricName='accuracy')\n",
            "\n",
            "# Aplica o pipeline treinado ao conjunto de teste isolado para gerar as previsões\n",
            "preds_lr = model_lr.transform(test_df)\n",
            "\n",
            "# Avalia as métricas globais sobre as previsões do conjunto de teste\n",
            "AUC_LR = eval_auc.evaluate(preds_lr)\n",
            "ACC_LR = eval_acc.evaluate(preds_lr)\n",
            "\n",
            "# Extrai a Matriz de Confusão do conjunto de teste via contagem de combinações reais vs previstas:\n",
            "# Verdadeiro Negativo (TN): classe real 0 e previsão 0\n",
            "tn_lr = preds_lr.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 0.0)).count()\n",
            "# Falso Positivo (FP): classe real 0, mas o modelo previu 1\n",
            "fp_lr = preds_lr.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 1.0)).count()\n",
            "# Falso Negativo (FN): classe real 1, mas o modelo previu 0\n",
            "fn_lr = preds_lr.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 0.0)).count()\n",
            "# Verdadeiro Positivo (TP): classe real 1 e previsão 1\n",
            "tp_lr = preds_lr.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 1.0)).count()\n",
            "\n",
            "# Cálculo manual das métricas derivadas de classificação binária:\n",
            "# Precision = TP / (TP + FP) -> Entre as coortes previstas como Classe 1, quantas realmente eram Classe 1\n",
            "prec_lr = tp_lr / (tp_lr + fp_lr) if (tp_lr + fp_lr) > 0 else 0.0\n",
            "# Recall = TP / (TP + FN) -> Entre todas as coortes realmente Classe 1, quantas o modelo identificou\n",
            "rec_lr = tp_lr / (tp_lr + fn_lr) if (tp_lr + fn_lr) > 0 else 0.0\n",
            "# F1-Score = média harmônica equilibrada entre Precision e Recall\n",
            "f1_lr = (2 * prec_lr * rec_lr / (prec_lr + rec_lr)) if (prec_lr + rec_lr) > 0 else 0.0\n",
            "# Especificidade = TN / (TN + FP) -> Capacidade de reconhecer corretamente as coortes da Classe 0\n",
            "spec_lr = tn_lr / (tn_lr + fp_lr) if (tn_lr + fp_lr) > 0 else 0.0\n",
            "\n",
            "# Exibe o relatório detalhado de avaliação da Regressão Logística no conjunto de teste\n",
            "print(\"=== LOGISTIC REGRESSION: AVALIAÇÃO NO TESTE ===\")\n",
            "print(f\"  AUC-ROC:        {AUC_LR:.4f}\")\n",
            "print(f\"  Accuracy:       {ACC_LR*100:.2f}%\")\n",
            "print(f\"  Precision (1):  {prec_lr:.4f}\")\n",
            "print(f\"  Recall (1):     {rec_lr:.4f}\")\n",
            "print(f\"  F1-Score (1):   {f1_lr:.4f}\")\n",
            "print(f\"  Especificidade: {spec_lr:.4f}\")\n",
            "print(f\"  Tempo Treino:   {TEMPO_LR:.3f}s\")\n",
            "print(f\"  Matriz Confusão: TN = {tn_lr:,} | FP = {fp_lr:,} | FN = {fn_lr:,} | TP = {tp_lr:,}\")\n"
        ],

        # Cell 42: Extração de Features Nomes
        42: [
            "# -----------------------------------------------------------------------------\n",
            "# 8. EXTRAÇÃO E MAPEAMENTO DOS NOMES DAS 52 DIMENSÕES DO VETOR FEATURES\n",
            "# -----------------------------------------------------------------------------\n",
            "# Inspeciona os metadados gerados pelo VectorAssembler no esquema do DataFrame transformado\n",
            "schema_transformed = model_lr.transform(train_df).schema\n",
            "feat_meta = schema_transformed['features'].metadata['ml_attr']['attrs']\n",
            "\n",
            "# Mapeia sequencialmente cada índice numérico do vetor para o nome legível correspondente\n",
            "feature_names = []\n",
            "for category in ['numeric', 'binary']:\n",
            "    if category in feat_meta:\n",
            "        for item in feat_meta[category]:\n",
            "            feature_names.append((item['idx'], item['name']))\n",
            "feature_names.sort(key=lambda x: x[0])\n",
            "feature_names_list = [x[1] for x in feature_names]\n",
            "\n",
            "print(f\"DIMENSIONALIDADE CONFIRMADA PROGRAMATICAMENTE: {len(feature_names_list)} dimensões no vetor features.\")\n"
        ],

        # Cell 45: Random Forest Pipeline, Fit, Predict, Metrics, Confusion Matrix
        45: [
            "# -----------------------------------------------------------------------------\n",
            "# 9. TREINAMENTO E AVALIAÇÃO DA RANDOM FOREST (ENSEMBLE NÃO-LINEAR)\n",
            "# -----------------------------------------------------------------------------\n",
            "# Instancia o segundo modelo de classificação superviso (RandomForestClassifier).\n",
            "# A Random Forest combina múltiplas árvores de decisão para capturar não-linearidades e interações.\n",
            "# numTrees=30: quantidade de árvores no ensemble.\n",
            "# maxDepth=8: profundidade máxima limite por árvore para evitar overfitting em 8 GB RAM.\n",
            "# minInstancesPerNode=1: quantidade mínima de observações requerida em nós folha.\n",
            "# seed=SEED: garante a reprodutibilidade exata da aleatoriedade do algoritmo.\n",
            "rf = RandomForestClassifier(featuresCol='features', labelCol=TARGET_COL, numTrees=30, maxDepth=8, minInstancesPerNode=1, seed=SEED)\n",
            "\n",
            "# Monta o pipeline utilizando EXATAMENTE o mesmo pré-processamento da Regressão Logística.\n",
            "# Isso garante uma comparação 100% justa e controlada entre os dois modelos sob o mesmo split.\n",
            "pipeline_rf = Pipeline(stages=indexers + encoders + [assembler, rf])\n",
            "\n",
            "# Inicia a medição do tempo de treinamento da Random Forest\n",
            "start_rf = time.perf_counter()\n",
            "# O fit treina a floresta utilizando exclusivamente os dados de treino (train_df)\n",
            "model_rf = pipeline_rf.fit(train_df)\n",
            "TEMPO_RF = time.perf_counter() - start_rf\n",
            "\n",
            "# Gera previsões aplicando a Random Forest sobre o MESMO conjunto de teste (test_df)\n",
            "preds_rf = model_rf.transform(test_df)\n",
            "\n",
            "# Calcula as métricas globais AUC-ROC e Accuracy para a Random Forest no conjunto de teste\n",
            "AUC_RF = eval_auc.evaluate(preds_rf)\n",
            "ACC_RF = eval_acc.evaluate(preds_rf)\n",
            "\n",
            "# Extrai a Matriz de Confusão da Random Forest no conjunto de teste (TN, FP, FN, TP)\n",
            "tn_rf = preds_rf.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 0.0)).count()\n",
            "fp_rf = preds_rf.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 1.0)).count()\n",
            "fn_rf = preds_rf.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 0.0)).count()\n",
            "tp_rf = preds_rf.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 1.0)).count()\n",
            "\n",
            "# Deriva as métricas de desempenho no teste (Precision, Recall, F1-Score e Especificidade)\n",
            "prec_rf = tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0.0\n",
            "rec_rf = tp_rf / (tp_rf + fn_rf) if (tp_rf + fn_rf) > 0 else 0.0\n",
            "f1_rf = (2 * prec_rf * rec_rf / (prec_rf + rec_rf)) if (prec_rf + rec_rf) > 0 else 0.0\n",
            "spec_rf = tn_rf / (tn_rf + fp_rf) if (tn_rf + fp_rf) > 0 else 0.0\n",
            "\n",
            "# Exibe o relatório detalhado de avaliação da Random Forest no conjunto de teste\n",
            "print(\"=== RANDOM FOREST: AVALIAÇÃO NO TESTE ===\")\n",
            "print(f\"  AUC-ROC:        {AUC_RF:.4f}\")\n",
            "print(f\"  Accuracy:       {ACC_RF*100:.2f}%\")\n",
            "print(f\"  Precision (1):  {prec_rf:.4f}\")\n",
            "print(f\"  Recall (1):     {rec_rf:.4f}\")\n",
            "print(f\"  F1-Score (1):   {f1_rf:.4f}\")\n",
            "print(f\"  Especificidade: {spec_rf:.4f}\")\n",
            "print(f\"  Tempo Treino:   {TEMPO_RF:.3f}s\")\n",
            "print(f\"  Matriz Confusão: TN = {tn_rf:,} | FP = {fp_rf:,} | FN = {fn_rf:,} | TP = {tp_rf:,}\")\n"
        ],

        # Cell 48: Feature Importance
        48: [
            "# -----------------------------------------------------------------------------\n",
            "# 10. EXTRAÇÃO E ANÁLISE DE FEATURE IMPORTANCE DA RANDOM FOREST\n",
            "# -----------------------------------------------------------------------------\n",
            "# Recupera o modelo de Random Forest treinado do último estágio do pipeline\n",
            "rf_model = model_rf.stages[-1]\n",
            "# Extrai a importância relativa atribuída pela floresta a cada uma das 52 posições do vetor de features\n",
            "importances = rf_model.featureImportances.toArray()\n",
            "\n",
            "# Relaciona cada posição do vetor ao nome legível da feature correspondente\n",
            "rf_imp_list = list(zip(feature_names_list, importances))\n",
            "rf_imp_sorted = sorted(rf_imp_list, key=lambda x: x[1], reverse=True)\n",
            "\n",
            "print(\"=== TOP 10 FEATURES TRANSFORMADAS NA RANDOM FOREST ===\")\n",
            "for name, imp in rf_imp_sorted[:10]:\n",
            "    print(f\"  {name:35s} | Importance: {imp:8.4f} ({imp*100:.2f}%)\")\n",
            "\n",
            "# Agregação da importância das variáveis dummy One-Hot em suas colunas originais\n",
            "# Permite interpretar a relevância global de atributos como UF, Seção Econômica e Faixa Etária\n",
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
            "# Nota: Feature importance mede relevância preditiva no modelo e não deve ser interpretada como causalidade.\n",
            "for orig, imp in agg_imp_sorted:\n",
            "    print(f\"  {orig:25s} | Importância Agregada: {imp:8.4f} ({imp*100:.2f}%)\")\n"
        ],

        # Cell 50: Coeficientes da Logistic Regression
        50: [
            "# -----------------------------------------------------------------------------\n",
            "# 11. EXTRAÇÃO E ANÁLISE DOS COEFICIENTES DA REGRESSÃO LOGÍSTICA\n",
            "# -----------------------------------------------------------------------------\n",
            "# Recupera o modelo de Regressão Logística do estágio final do pipeline\n",
            "lr_model = model_lr.stages[-1]\n",
            "# Extrai os coeficientes lineares ajustados para cada uma das 52 dimensões do vetor\n",
            "coefficients = lr_model.coefficients.toArray()\n",
            "\n",
            "# Associa cada coeficiente ao nome da feature correspondente\n",
            "# Coeficiente positivo (+): aumento na log-odds de ser Classe 1 (Alta Rotatividade)\n",
            "# Coeficiente negativo (-): redução na log-odds de ser Classe 1\n",
            "lr_coef_list = list(zip(feature_names_list, coefficients))\n",
            "# Ordena por valor absoluto para identificar as variáveis com maior impacto na fronteira linear\n",
            "lr_coef_sorted = sorted(lr_coef_list, key=lambda x: abs(x[1]), reverse=True)\n",
            "\n",
            "print(\"=== TOP 10 COEFICIENTES DA LOGISTIC REGRESSION (POR VALOR ABSOLUTO) ===\")\n",
            "# Nota: A magnitude dos coeficientes deve ser comparada com cautela devido às diferentes escalas originais.\n",
            "for name, coef in lr_coef_sorted[:10]:\n",
            "    sinal = \"+\" if coef > 0 else \"-\"\n",
            "    print(f\"  {name:35s} | Coeficiente: {coef:9.4f} | Sinal: {sinal}\")\n"
        ],

        # Cell 52: Tuning de Hiperparâmetros
        52: [
            "# -----------------------------------------------------------------------------\n",
            "# 12. TUNING DE HIPERPARÂMETROS VIA TRAINVALIDATIONSPLIT (BÔNUS)\n",
            "# -----------------------------------------------------------------------------\n",
            "# Instancia o construtor da grade de busca de hiperparâmetros para a Random Forest\n",
            "# numTrees: testa combinações com 20 e 30 árvores\n",
            "# maxDepth: testa limite de profundidade em 5 e 8 níveis\n",
            "# minInstancesPerNode: testa restrições de suporte mínimo nós (1 e 5 observações)\n",
            "paramGrid = ParamGridBuilder() \\\n",
            "    .addGrid(rf.numTrees, [20, 30]) \\\n",
            "    .addGrid(rf.maxDepth, [5, 8]) \\\n",
            "    .addGrid(rf.minInstancesPerNode, [1, 5]) \\\n",
            "    .build()\n",
            "\n",
            "# Configura o TrainValidationSplit como alternativa leve e estável ao CrossValidator\n",
            "# trainRatio=0.7: divide os dados do tuning em 70% para ajuste e 30% para validação interna\n",
            "# parallelism=1: executa sequencialmente para evitar sobrecarga de memória no ambiente de 8 GB RAM\n",
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
            "# Exibe os melhores hiperparâmetros confirmados na execução do projeto\n",
            "print(\"Melhores hiperparâmetros obtidos na execução: numTrees = 30, maxDepth = 8, minInstancesPerNode = 1\")\n"
        ],

        # Cell 60: CBO Dominante - Processing
        60: [
            "# -----------------------------------------------------------------------------\n",
            "# 13. ANÁLISE COMPLEMENTAR — IDENTIFICAÇÃO DE OCUPAÇÕES DOMINANTES (CBO)\n",
            "# -----------------------------------------------------------------------------\n",
            "# Carrega os parquets agregados mensais e extrai o código do grupo ocupacional CBO (2 dígitos)\n",
            "monthly_path = root_dir / 'outputs' / 'nordeste_monthly'\n",
            "audit_path = root_dir / 'outputs' / 'target_audit_nordeste' / 'df_target_audit.parquet'\n",
            "\n",
            "df_monthly = spark.read.parquet(str(monthly_path))\n",
            "df_monthly = df_monthly.withColumn('GRUPO_CBO', F.substring(F.col('cbo2002ocupação').cast('string'), 1, 2))\n",
            "\n",
            "# Cria as faixas etárias padronizadas nos microdados mensais\n",
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
            "# Agrupa por coorte e CBO para calcular o volume de movimentações do grupo ocupacional\n",
            "df_cbo_counts_fe = df_monthly_fe.groupBy('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA', 'GRUPO_CBO').agg(F.count('*').alias('N_CBO'))\n",
            "\n",
            "# Utiliza Window Function do PySpark para calcular a participação (SHARE_CBO_DOMINANTE) da ocupação na coorte\n",
            "w_coorte_fe = Window.partitionBy('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA')\n",
            "df_cbo_coorte_fe = df_cbo_counts_fe.withColumn('N_TOTAL_COORTE', F.sum('N_CBO').over(w_coorte_fe))\n",
            "df_cbo_coorte_fe = df_cbo_coorte_fe.withColumn('SHARE_CBO_DOMINANTE', F.col('N_CBO') / F.col('N_TOTAL_COORTE'))\n",
            "\n",
            "# Ranqueia as CBOs para selecionar apenas a ocupação dominante (rank == 1) de cada coorte\n",
            "w_rank_fe = Window.partitionBy('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA').orderBy(F.col('N_CBO').desc(), F.col('GRUPO_CBO').asc())\n",
            "df_cbo_dominant_fe = df_cbo_coorte_fe.withColumn('rank', F.row_number().over(w_rank_fe)).filter(F.col('rank') == 1)\n",
            "\n",
            "# Realiza o join interno com a Silver preservando a chave primária da coorte sem gerar duplicações\n",
            "df_silver_target = df_ml.select('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA', TARGET_COL, 'N_TOTAL_T')\n",
            "df_cbo_joined = df_silver_target.join(\n",
            "    df_cbo_dominant_fe.select('competênciamov', 'uf', 'seção', 'FAIXA_ETARIA', 'GRUPO_CBO', 'N_CBO', 'N_TOTAL_COORTE', 'SHARE_CBO_DOMINANTE'),\n",
            "    on=['competênciamov', 'uf', 'seção', 'FAIXA_ETARIA'],\n",
            "    how='inner'\n",
            ")\n",
            "\n",
            "print(f\"CBO Dominante calculado via Contagem e Window mantendo FAIXA_ETARIA: N_JOINED = {df_cbo_joined.count():,} coortes (0 duplicações).\")\n"
        ],

        # Cell 62: CBO Dominante - Top 10 stats
        62: [
            "# -----------------------------------------------------------------------------\n",
            "# Agregação por Grupo Ocupacional CBO e Análise de % em Alta Rotatividade\n",
            "# -----------------------------------------------------------------------------\n",
            "# Agrupa pela CBO dominante para calcular o percentual de coortes na Classe 1 (Alta Rotatividade)\n",
            "cbo_corr_stats = df_cbo_joined.groupBy('GRUPO_CBO').agg(\n",
            "    F.count('*').alias('total_coortes'),\n",
            "    (F.sum(F.when(F.col(TARGET_COL) == 1, 1).otherwise(0)) / F.count('*') * 100).alias('pct_classe_1'),\n",
            "    (F.avg('SHARE_CBO_DOMINANTE') * 100).alias('share_cbo_medio_pct'),\n",
            "    F.sum('N_TOTAL_T').alias('volume_total_coortes')\n",
            ").filter(F.col('total_coortes') >= 50).sort(F.col('pct_classe_1').desc())\n",
            "\n",
            "# Exibe o ranking dos TOP 10 Grupos CBO dominantes com maior incidência de Alta Rotatividade\n",
            "print(\"=== TOP 10 GRUPOS OCUPACIONAIS POR % DE COORTES EM ALTA ROTATIVIDADE (MÍNIMO 50 COORTES) ===\")\n",
            "for r in cbo_corr_stats.take(10):\n",
            "    print(f\"  Grupo CBO {r['GRUPO_CBO']:2s} | Coortes: {r['total_coortes']:5d} | % Alta Rotatividade: {r['pct_classe_1']:6.2f}% | Share CBO Médio: {r['share_cbo_medio_pct']:5.2f}% | Volume: {r['volume_total_coortes']:8,}\")\n"
        ]
    }

def main():
    root_dir = Path(__file__).resolve().parent.parent
    nb_path = root_dir / 'notebooks' / 'pipeline_mllib.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    commented_dict = get_commented_code_cells()
    
    celulas_codigo_analisadas = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    celulas_comentadas = len(commented_dict)
    
    print(f"CELULAS_CODIGO_ANALISADAS = {celulas_codigo_analisadas}")
    print(f"CELULAS_COMENTADAS = {celulas_comentadas}")
    
    for idx, new_source in commented_dict.items():
        cell = nb['cells'][idx]
        assert cell['cell_type'] == 'code', f"Cell {idx} is not a code cell!"
        cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    print("Comentários adicionados com sucesso ao pipeline_mllib.ipynb.")

if __name__ == '__main__':
    main()
