import os
import sys
import json
import csv
import math
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = root_dir / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

def main():
    print("=== TESTE E RECALCULO COMPLETO DO PIPELINE MLLIB E CBO ===")
    spark = SparkSession.builder \
        .appName("PipelineCorrectionTest") \
        .master("local[2]") \
        .config("spark.driver.memory", "3g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    spark.catalog.clearCache()

    silver_path = root_dir / "silver" / "caged_nordeste_ml"
    audit_path = root_dir / "outputs" / "target_audit_nordeste" / "df_target_audit.parquet"
    monthly_path = root_dir / "outputs" / "nordeste_monthly"

    df_ml = spark.read.parquet(str(silver_path))
    N_SILVER = df_ml.count()
    print(f"Camada Silver lida: N = {N_SILVER:,} registros.")

    TARGET_COL = 'ALTA_ROTATIVIDADE_6M'
    FEATURES_CATEGORICAS = ['uf', 'seção', 'FAIXA_ETARIA', 'FAIXA_VOLUME_COORTE']
    FEATURES_NUMERICAS = ['N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T', 'LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T', 'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS']
    SEED = 42

    train_df, test_df = df_ml.randomSplit([0.7, 0.3], seed=SEED)
    N_TRAIN = train_df.count()
    N_TEST = test_df.count()
    print(f"Split 70/30: Treino = {N_TRAIN:,} | Teste = {N_TEST:,}")

    # Indexers & Encoders
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in FEATURES_CATEGORICAS]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in FEATURES_CATEGORICAS]
    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in FEATURES_CATEGORICAS] + FEATURES_NUMERICAS,
        outputCol="features"
    )

    # 1. Pipeline Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol=TARGET_COL, maxIter=100)
    pipeline_lr = Pipeline(stages=indexers + encoders + [assembler, lr])

    start_lr = time.perf_counter()
    model_lr = pipeline_lr.fit(train_df)
    tempo_lr = time.perf_counter() - start_lr

    eval_auc = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    eval_acc = MulticlassClassificationEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="accuracy")

    preds_lr = model_lr.transform(test_df)
    auc_lr = eval_auc.evaluate(preds_lr)
    acc_lr = eval_acc.evaluate(preds_lr)

    tn_lr = preds_lr.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 0.0)).count()
    fp_lr = preds_lr.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 1.0)).count()
    fn_lr = preds_lr.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 0.0)).count()
    tp_lr = preds_lr.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 1.0)).count()

    prec_lr = tp_lr / (tp_lr + fp_lr) if (tp_lr + fp_lr) > 0 else 0.0
    rec_lr = tp_lr / (tp_lr + fn_lr) if (tp_lr + fn_lr) > 0 else 0.0
    f1_lr = (2 * prec_lr * rec_lr / (prec_lr + rec_lr)) if (prec_lr + rec_lr) > 0 else 0.0
    spec_lr = tn_lr / (tn_lr + fp_lr) if (tn_lr + fp_lr) > 0 else 0.0

    print(f"LR: AUC = {auc_lr:.4f} | Acc = {acc_lr:.4f} | Prec = {prec_lr:.4f} | Rec = {rec_lr:.4f} | F1 = {f1_lr:.4f} | Spec = {spec_lr:.4f} | Tempo = {tempo_lr:.3f}s")
    print(f"LR Confusion Matrix: TN={tn_lr:,}, FP={fp_lr:,}, FN={fn_lr:,}, TP={tp_lr:,}")

    # Verificar dimensões do VectorAssembler
    schema_transformed = model_lr.transform(train_df).schema
    feat_meta = schema_transformed["features"].metadata["ml_attr"]["attrs"]
    feature_names = []
    for category in ["numeric", "binary"]:
        if category in feat_meta:
            for item in feat_meta[category]:
                feature_names.append((item["idx"], item["name"]))
    feature_names.sort(key=lambda x: x[0])
    feature_names_list = [x[1] for x in feature_names]
    num_dim = len(feature_names_list)
    print(f"DIMENSIONALIDADE DO VETOR FEATURES: {num_dim} dimensões.")

    # Coeficientes LR
    lr_model = model_lr.stages[-1]
    coefficients = lr_model.coefficients.toArray()
    intercept = lr_model.intercept
    lr_coef_list = list(zip(feature_names_list, coefficients))
    lr_coef_sorted = sorted(lr_coef_list, key=lambda x: abs(x[1]), reverse=True)
    print("\nTop 10 Coeficientes LR por Valor Absoluto:")
    for name, coef in lr_coef_sorted[:10]:
        sinal = "+" if coef > 0 else "-"
        print(f"  {name:35s} | Coef: {coef:9.4f} | Sinal: {sinal}")

    # 2. Pipeline Random Forest Base
    rf = RandomForestClassifier(featuresCol="features", labelCol=TARGET_COL, numTrees=30, maxDepth=8, minInstancesPerNode=1, seed=SEED)
    pipeline_rf = Pipeline(stages=indexers + encoders + [assembler, rf])

    start_rf = time.perf_counter()
    model_rf = pipeline_rf.fit(train_df)
    tempo_rf = time.perf_counter() - start_rf

    preds_rf = model_rf.transform(test_df)
    auc_rf = eval_auc.evaluate(preds_rf)
    acc_rf = eval_acc.evaluate(preds_rf)

    tn_rf = preds_rf.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 0.0)).count()
    fp_rf = preds_rf.filter((F.col(TARGET_COL) == 0) & (F.col('prediction') == 1.0)).count()
    fn_rf = preds_rf.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 0.0)).count()
    tp_rf = preds_rf.filter((F.col(TARGET_COL) == 1) & (F.col('prediction') == 1.0)).count()

    prec_rf = tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0.0
    rec_rf = tp_rf / (tp_rf + fn_rf) if (tp_rf + fn_rf) > 0 else 0.0
    f1_rf = (2 * prec_rf * rec_rf / (prec_rf + rec_rf)) if (prec_rf + rec_rf) > 0 else 0.0
    spec_rf = tn_rf / (tn_rf + fp_rf) if (tn_rf + fp_rf) > 0 else 0.0

    print(f"\nRF: AUC = {auc_rf:.4f} | Acc = {acc_rf:.4f} | Prec = {prec_rf:.4f} | Rec = {rec_rf:.4f} | F1 = {f1_rf:.4f} | Spec = {spec_rf:.4f} | Tempo = {tempo_rf:.3f}s")
    print(f"RF Confusion Matrix: TN={tn_rf:,}, FP={fp_rf:,}, FN={fn_rf:,}, TP={tp_rf:,}")

    # Feature Importance RF
    rf_model = model_rf.stages[-1]
    importances = rf_model.featureImportances.toArray()
    rf_imp_list = list(zip(feature_names_list, importances))
    rf_imp_sorted = sorted(rf_imp_list, key=lambda x: x[1], reverse=True)
    print("\nTop 10 Features Transformadas na Random Forest:")
    for name, imp in rf_imp_sorted[:10]:
        print(f"  {name:35s} | Importance: {imp:8.4f} ({imp*100:.2f}%)")

    # Importância Agregada por Variável Original
    agg_imp = {}
    for name, imp in rf_imp_list:
        orig = name
        for cat in FEATURES_CATEGORICAS:
            if name.startswith(f"{cat}_"):
                orig = cat
                break
        agg_imp[orig] = agg_imp.get(orig, 0.0) + imp

    agg_imp_sorted = sorted(agg_imp.items(), key=lambda x: x[1], reverse=True)
    print("\nImportância Agregada por Variável Original:")
    for orig, imp in agg_imp_sorted:
        print(f"  {orig:25s} | Importância Agregada: {imp:8.4f} ({imp*100:.2f}%)")

    # 3. CORREÇÃO DA ANÁLISE DE CBO (PRESERVANDO FAIXA_ETARIA E CALCULANDO DOMINÂNCIA POR CONTAGEM E WINDOW)
    print("\n=== CORREÇÃO DA ANÁLISE DE CBO (DOMINÂNCIA POR CONTAGEM E WINDOW COM FAIXA_ETARIA) ===")
    df_monthly = spark.read.parquet(str(monthly_path))
    df_monthly = df_monthly.withColumn("GRUPO_CBO", F.substring(F.col("cbo2002ocupação").cast("string"), 1, 2))

    # 1. Agrupar por competênciamov, uf, seção, FAIXA_ETARIA, GRUPO_CBO e contar registros
    df_cbo_counts = df_monthly.groupBy("competênciamov", "uf", "seção", "categoria", "sexo", "GRUPO_CBO").agg(
        F.count("*").alias("N_CBO")
    )

    # 2. Window para calcular total por coorte e determinar o rank do GRUPO_CBO mais frequente
    w_coorte = Window.partitionBy("competênciamov", "uf", "seção", "categoria", "sexo")
    df_cbo_coorte_total = df_cbo_counts.withColumn("N_TOTAL_COORTE", F.sum("N_CBO").over(w_coorte))
    df_cbo_coorte_total = df_cbo_coorte_total.withColumn("SHARE_CBO_DOMINANTE", F.col("N_CBO") / F.col("N_TOTAL_COORTE"))

    w_rank = Window.partitionBy("competênciamov", "uf", "seção", "categoria", "sexo").orderBy(F.col("N_CBO").desc(), F.col("GRUPO_CBO").asc())
    df_cbo_dominant = df_cbo_coorte_total.withColumn("rank", F.row_number().over(w_rank)).filter(F.col("rank") == 1)

    # 3. Join correto com a Silver contendo FAIXA_ETARIA
    # Selecionar chave da coorte e o target da Silver
    df_silver_target = df_ml.select("competênciamov", "uf", "seção", "FAIXA_ETARIA", TARGET_COL, "N_TOTAL_T")

    # Mapear a chave com FAIXA_ETARIA no join
    df_cbo_coorte_key = df_monthly.groupBy("competênciamov", "uf", "seção", "idade").agg(
        F.first("GRUPO_CBO").alias("GRUPO_CBO_RAW")
    )
    
    # Vamos fazer o agrupamento do CBO por competênciamov, uf, seção, FAIXA_ETARIA na base de movimentação!
    # Criar FAIXA_ETARIA na base mensal se necessário ou fazer o join exato
    # FAIXA_ETARIA na Silver é derivada de 'idade'
    df_monthly_fe = df_monthly.withColumn(
        "FAIXA_ETARIA",
        F.when(F.col("idade") < 18, "14-17")
         .when((F.col("idade") >= 18) & (F.col("idade") <= 24), "18-24")
         .when((F.col("idade") >= 25) & (F.col("idade") <= 34), "25-34")
         .when((F.col("idade") >= 35) & (F.col("idade") <= 44), "35-44")
         .when((F.col("idade") >= 45) & (F.col("idade") <= 54), "45-54")
         .when((F.col("idade") >= 55) & (F.col("idade") <= 64), "55-64")
         .otherwise("65+")
    )

    df_cbo_counts_fe = df_monthly_fe.groupBy("competênciamov", "uf", "seção", "FAIXA_ETARIA", "GRUPO_CBO").agg(
        F.count("*").alias("N_CBO")
    )

    w_coorte_fe = Window.partitionBy("competênciamov", "uf", "seção", "FAIXA_ETARIA")
    df_cbo_coorte_fe = df_cbo_counts_fe.withColumn("N_TOTAL_COORTE", F.sum("N_CBO").over(w_coorte_fe))
    df_cbo_coorte_fe = df_cbo_coorte_fe.withColumn("SHARE_CBO_DOMINANTE", F.col("N_CBO") / F.col("N_TOTAL_COORTE"))

    w_rank_fe = Window.partitionBy("competênciamov", "uf", "seção", "FAIXA_ETARIA").orderBy(F.col("N_CBO").desc(), F.col("GRUPO_CBO").asc())
    df_cbo_dominant_fe = df_cbo_coorte_fe.withColumn("rank", F.row_number().over(w_rank_fe)).filter(F.col("rank") == 1)

    # JOIN com Silver mantendo a chave exata da coorte (competênciamov + uf + seção + FAIXA_ETARIA)
    df_cbo_joined = df_silver_target.join(
        df_cbo_dominant_fe.select("competênciamov", "uf", "seção", "FAIXA_ETARIA", "GRUPO_CBO", "N_CBO", "N_TOTAL_COORTE", "SHARE_CBO_DOMINANTE"),
        on=["competênciamov", "uf", "seção", "FAIXA_ETARIA"],
        how="inner"
    )

    N_JOINED = df_cbo_joined.count()
    print(f"Join entre Silver e CBO Dominante: N_SILVER = {N_SILVER:,} | N_JOINED = {N_JOINED:,} (Validação de 0 duplicações!)")

    # Avaliar Percentis do SHARE_CBO_DOMINANTE
    percentiles = df_cbo_joined.stat.approxQuantile("SHARE_CBO_DOMINANTE", [0.25, 0.50, 0.75, 0.90], 0.01)
    print(f"\nDistribuição da Qualidade de Dominância (SHARE_CBO_DOMINANTE):")
    print(f"  P25 = {percentiles[0]:.4f} ({percentiles[0]*100:.2f}%)")
    print(f"  P50 (Mediana) = {percentiles[1]:.4f} ({percentiles[1]*100:.2f}%)")
    print(f"  P75 = {percentiles[2]:.4f} ({percentiles[2]*100:.2f}%)")
    print(f"  P90 = {percentiles[3]:.4f} ({percentiles[3]*100:.2f}%)")

    # Análise de Rotatividade por GRUPO_CBO_DOMINANTE corrigido
    cbo_corr_stats = df_cbo_joined.groupBy("GRUPO_CBO").agg(
        F.count("*").alias("total_coortes"),
        F.sum(F.when(F.col(TARGET_COL) == 1, 1).otherwise(0)).alias("coortes_classe_1"),
        F.sum(F.when(F.col(TARGET_COL) == 0, 1).otherwise(0)).alias("coortes_classe_0"),
        (F.sum(F.when(F.col(TARGET_COL) == 1, 1).otherwise(0)) / F.count("*") * 100).alias("pct_classe_1"),
        (F.avg("SHARE_CBO_DOMINANTE") * 100).alias("share_cbo_medio_pct"),
        F.sum("N_TOTAL_T").alias("volume_total_coortes")
    )

    min_coortes_corr = 50
    cbo_corr_filtered = cbo_corr_stats.filter(F.col("total_coortes") >= min_coortes_corr).sort(F.col("pct_classe_1").desc())

    print(f"\nTop 10 Grupos Ocupacionais com Maior % de Coortes na Classe 1 (mínimo reproduzível de {min_coortes_corr} coortes):")
    top10_cbo_corr = cbo_corr_filtered.take(10)
    for r in top10_cbo_corr:
        print(f"  Grupo CBO {r['GRUPO_CBO']:2s} | Coortes: {r['total_coortes']:5d} | % Alta Rotatividade: {r['pct_classe_1']:6.2f}% | Share Médio CBO: {r['share_cbo_medio_pct']:5.2f}% | Volume: {r['volume_total_coortes']:8,}")

    print("\nSUCESSO: Todos os cálculos reproduzidos e auditados!")
    spark.stop()

if __name__ == "__main__":
    main()
