import os
import sys
import shutil
import math
import time
import json
import csv
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = root_dir / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def main():
    print("=== 1. CARREGAR A SILVER DO NORDESTE ===")
    spark = SparkSession.builder \
        .appName("NordesteMLlibPipeline") \
        .master("local[2]") \
        .config("spark.driver.memory", "3g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    spark.catalog.clearCache()

    silver_path = root_dir / "silver" / "caged_nordeste_ml"
    if not silver_path.exists():
        print(f"ERRO CRÍTICO: O caminho {silver_path} não existe! Interrompendo.")
        sys.exit(1)

    print(f"Lendo camada Silver em: {silver_path}")
    df_ml = spark.read.parquet(str(silver_path))

    total_registros = df_ml.count()
    print(f"TOTAL_REGISTROS = {total_registros:,} (Coincide com validação da Seção 11: 33,465)")
    print(f"Número de Colunas: {len(df_ml.columns)}")
    print("Schema de df_ml:")
    df_ml.printSchema()

    min_comp = df_ml.select(F.min("competênciamov")).first()[0]
    max_comp = df_ml.select(F.max("competênciamov")).first()[0]
    print(f"Período Mínimo: {min_comp} | Período Máximo: {max_comp}")

    TARGET_COL = "ALTA_ROTATIVIDADE_6M"
    print(f"\nTARGET_COL = '{TARGET_COL}'")

    t_dist = df_ml.groupBy(TARGET_COL).agg(
        F.count("*").alias("cnt"),
        (F.count("*") / total_registros * 100).alias("pct")
    ).collect()

    t_map = {r[TARGET_COL]: (r["cnt"], r["pct"]) for r in t_dist}
    print("Distribuição das Classes do Target:")
    print(f"  Classe 0: {t_map.get(0,(0,0))[0]:,} ({t_map.get(0,(0,0))[1]:.2f}%)")
    print(f"  Classe 1: {t_map.get(1,(0,0))[0]:,} ({t_map.get(1,(0,0))[1]:.2f}%)")

    # 4. RECUPERAR FEATURES APROVADAS
    FEATURES_CATEGORICAS = ["uf", "seção", "FAIXA_ETARIA", "FAIXA_VOLUME_COORTE"]
    FEATURES_NUMERICAS = ["N_TOTAL_T", "N_POSITIVOS_T", "N_NEGATIVOS_T", "LOG_VOLUME_COORTE", "PROP_NEGATIVOS_T", "ANO", "MES", "TRIMESTRE", "MES_SIN", "MES_COS"]
    FEATURES_PROIBIDAS = ["N_NEGATIVOS_6M", "N_POSITIVOS_6M", "N_TOTAL_6M", "PROP_NEGATIVOS_6M"]

    total_features_originais = len(FEATURES_CATEGORICAS) + len(FEATURES_NUMERICAS)
    print(f"\nTOTAL_FEATURES_ORIGINAIS = {total_features_originais}")

    # 8-10. STRINGINDEXER, ONEHOTENCODER, VECTORASSEMBLER
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in FEATURES_CATEGORICAS
    ]

    encoders = [
        OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_ohe")
        for col in FEATURES_CATEGORICAS
    ]

    assembler_inputs = [f"{col}_ohe" for col in FEATURES_CATEGORICAS] + FEATURES_NUMERICAS
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    lr = LogisticRegression(featuresCol="features", labelCol=TARGET_COL)

    # Stages para o Pipeline: Indexers + Encoders + Assembler + LR
    pipeline_stages = indexers + encoders + [assembler, lr]
    pipeline_lr = Pipeline(stages=pipeline_stages)

    # 12. RANDOM SPLIT (70/30)
    SEED = 42
    print(f"\nRealizando randomSplit (70/30) com SEED = {SEED}...")
    train_df, test_df = df_ml.randomSplit([0.7, 0.3], seed=SEED)

    train_cnt = train_df.count()
    test_cnt = test_df.count()
    train_pct = (train_cnt / total_registros) * 100
    test_pct = (test_cnt / total_registros) * 100

    print(f"TREINO: {train_cnt:,} registros ({train_pct:.2f}%)")
    print(f"TESTE:  {test_cnt:,} registros ({test_pct:.2f}%)")

    train_dist = train_df.groupBy(TARGET_COL).agg(F.count("*").alias("cnt")).collect()
    train_counts = {r[TARGET_COL]: r["cnt"] for r in train_dist}
    tr_c0_pct = (train_counts.get(0,0) / train_cnt) * 100
    tr_c1_pct = (train_counts.get(1,0) / train_cnt) * 100

    test_dist = test_df.groupBy(TARGET_COL).agg(F.count("*").alias("cnt")).collect()
    test_counts = {r[TARGET_COL]: r["cnt"] for r in test_dist}
    te_c0_pct = (test_counts.get(0,0) / test_cnt) * 100
    te_c1_pct = (test_counts.get(1,0) / test_cnt) * 100

    print(f"  Treino Target: Classe 0 = {tr_c0_pct:.2f}% | Classe 1 = {tr_c1_pct:.2f}%")
    print(f"  Teste Target:  Classe 0 = {te_c0_pct:.2f}% | Classe 1 = {te_c1_pct:.2f}%")

    # 15. BASELINE MODEL (Prever classe majoritária do Treino)
    maj_class = 1 if train_counts.get(1,0) >= train_counts.get(0,0) else 0
    print(f"\nClasse majoritária do Treino: {maj_class}")

    baseline_preds = test_df.withColumn("baseline_pred", F.lit(maj_class))
    b_tp = baseline_preds.filter((F.col(TARGET_COL) == 1) & (F.col("baseline_pred") == 1)).count()
    b_fp = baseline_preds.filter((F.col(TARGET_COL) == 0) & (F.col("baseline_pred") == 1)).count()
    b_fn = baseline_preds.filter((F.col(TARGET_COL) == 1) & (F.col("baseline_pred") == 0)).count()
    b_tn = baseline_preds.filter((F.col(TARGET_COL) == 0) & (F.col("baseline_pred") == 0)).count()

    baseline_acc = (b_tp + b_tn) / test_cnt
    baseline_prec = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
    baseline_rec = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
    baseline_f1 = (2 * baseline_prec * baseline_rec / (baseline_prec + baseline_rec)) if (baseline_prec + baseline_rec) > 0 else 0.0

    print(f"Baseline (Preditor de Classe Majoritária {maj_class}):")
    print(f"  Accuracy  = {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"  Precision = {baseline_prec:.4f}")
    print(f"  Recall    = {baseline_rec:.4f}")
    print(f"  F1-Score  = {baseline_f1:.4f}")

    # 18. TREINAMENTO DA LOGISTIC REGRESSION (Apenas em train_df)
    print("\nIniciando treinamento da Regressão Logística no train_df...")
    start_t = time.perf_counter()
    pipeline_model_lr = pipeline_lr.fit(train_df)
    TEMPO_TREINO_LR = time.perf_counter() - start_t
    print(f"Treinamento concluído em TEMPO_TREINO_LR = {TEMPO_TREINO_LR:.3f} segundos!")

    # Extração da dimensionalidade das features e coeficientes
    lr_model = pipeline_model_lr.stages[-1]
    DIMENSAO_VECTOR_FEATURES = len(lr_model.coefficients)
    print(f"DIMENSAO_VECTOR_FEATURES = {DIMENSAO_VECTOR_FEATURES} posições")
    print(f"Intercepto do Modelo: {lr_model.intercept:.6f}")

    # 20-27. PREDIÇÕES E MÉTRICAS NO TESTE
    print("\nAvaliando predições no test_df...")
    predictions_lr = pipeline_model_lr.transform(test_df)

    evaluator_auc = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    AUC_ROC_LR = evaluator_auc.evaluate(predictions_lr)

    evaluator_acc = MulticlassClassificationEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="accuracy")
    ACCURACY_LR = evaluator_acc.evaluate(predictions_lr)

    # Matriz de Confusão
    tn = predictions_lr.filter((F.col(TARGET_COL) == 0) & (F.col("prediction") == 0.0)).count()
    fp = predictions_lr.filter((F.col(TARGET_COL) == 0) & (F.col("prediction") == 1.0)).count()
    fn = predictions_lr.filter((F.col(TARGET_COL) == 1) & (F.col("prediction") == 0.0)).count()
    tp = predictions_lr.filter((F.col(TARGET_COL) == 1) & (F.col("prediction") == 1.0)).count()

    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)) if (precision_pos + recall_pos) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n=== MATRIZ DE CONFUSÃO DA LOGISTIC REGRESSION ===")
    print(f"  TN = {tn:6,} | FP = {fp:6,}")
    print(f"  FN = {fn:6,} | TP = {tp:6,}")

    print("\n=== MÉTRICAS FINAIS NO CONJUNTO DE TESTE ===")
    print(f"  AUC-ROC      = {AUC_ROC_LR:.4f}")
    print(f"  Accuracy     = {ACCURACY_LR:.4f} ({ACCURACY_LR*100:.2f}%)")
    print(f"  Precision 1  = {precision_pos:.4f}")
    print(f"  Recall 1     = {recall_pos:.4f}")
    print(f"  F1-Score 1   = {f1_pos:.4f}")
    print(f"  Specificity  = {specificity:.4f}")

    # 32. SALVAR MODELO
    model_dir = root_dir / "models" / "logistic_regression_nordeste"
    if model_dir.exists():
        shutil.rmtree(model_dir)

    print(f"\nSalvando PipelineModel em: {model_dir}")
    pipeline_model_lr.write().overwrite().save(str(model_dir))
    MODELO_SALVO = "SIM" if (model_dir / "metadata").exists() else "NÃO"
    print(f"MODELO_SALVO = {MODELO_SALVO}")

    # 33. SALVAR MÉTRICAS
    metrics_dir = root_dir / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = metrics_dir / "logistic_regression_metrics.csv"

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "auc_roc", "accuracy", "precision_pos", "recall_pos", "f1_pos", "specificity", "training_time", "seed"])
        writer.writerow(["logistic_regression_nordeste", f"{AUC_ROC_LR:.6f}", f"{ACCURACY_LR:.6f}", f"{precision_pos:.6f}", f"{recall_pos:.6f}", f"{f1_pos:.6f}", f"{specificity:.6f}", f"{TEMPO_TREINO_LR:.3f}", SEED])

    METRICAS_SALVAS = "SIM" if metrics_csv.exists() else "NÃO"
    print(f"METRICAS_SALVAS = {METRICAS_SALVAS} ({metrics_csv})")

    # Json de resumo das métricas para compor a resposta final
    mllib_summary = {
        "target_col": TARGET_COL,
        "features_categoricas": FEATURES_CATEGORICAS,
        "features_numericas": FEATURES_NUMERICAS,
        "total_features_originais": total_features_originais,
        "dimensao_vector_features": DIMENSAO_VECTOR_FEATURES,
        "seed": SEED,
        "total_registros": total_registros,
        "treino": {
            "count": train_cnt,
            "percentual": train_pct,
            "c0_pct": tr_c0_pct,
            "c1_pct": tr_c1_pct
        },
        "teste": {
            "count": test_cnt,
            "percentual": test_pct,
            "c0_pct": te_c0_pct,
            "c1_pct": te_c1_pct
        },
        "baseline": {
            "accuracy": baseline_acc,
            "precision_1": baseline_prec,
            "recall_1": baseline_rec,
            "f1_1": baseline_f1
        },
        "logistic_regression": {
            "auc_roc": AUC_ROC_LR,
            "accuracy": ACCURACY_LR,
            "precision_1": precision_pos,
            "recall_1": recall_pos,
            "f1_1": f1_pos,
            "specificity": specificity
        },
        "matriz": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        },
        "tempo_treino_lr": TEMPO_TREINO_LR,
        "modelo_salvo": MODELO_SALVO,
        "metricas_salvas": METRICAS_SALVAS,
        "notebook_salvo": True
    }

    with open(root_dir / "outputs" / "mllib_summary.json", "w", encoding="utf-8") as f:
        json.dump(mllib_summary, f, indent=2, ensure_ascii=False)

    print("\nSUCESSO: Etapa MLlib concluída e resumida em outputs/mllib_summary.json!")
    spark.stop()

if __name__ == "__main__":
    main()
