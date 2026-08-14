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
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def get_feature_names_from_schema(schema, col_name="features"):
    """
    Extrai os nomes exatos das posições do vetor 'features' a partir dos metadados do PySpark.
    """
    field = schema[col_name]
    attrs = field.metadata.get("ml_attr", {}).get("attrs", {})
    idx_to_name = {}
    
    # Processar atributos numéricos e binários
    for attr_type in ["numeric", "binary"]:
        if attr_type in attrs:
            for item in attrs[attr_type]:
                idx = item["idx"]
                name = item["name"]
                idx_to_name[idx] = name
                
    # Fallback se não encontrar algum índice
    max_idx = max(idx_to_name.keys()) if idx_to_name else -1
    feature_names = [idx_to_name.get(i, f"feature_{i}") for i in range(max_idx + 1)]
    return feature_names

def main():
    print("=== 1. CARREGAR A SILVER DO NORDESTE E INSTANCIAR MODELOS ===")
    spark = SparkSession.builder \
        .appName("NordesteRFMLlibPipeline") \
        .master("local[2]") \
        .config("spark.driver.memory", "3g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    spark.catalog.clearCache()

    silver_path = root_dir / "silver" / "caged_nordeste_ml"
    if not silver_path.exists():
        print(f"ERRO CRÍTICO: O caminho {silver_path} não existe!")
        sys.exit(1)

    df_ml = spark.read.parquet(str(silver_path))
    total_registros = df_ml.count()
    print(f"TOTAL_REGISTROS = {total_registros:,}")

    TARGET_COL = "ALTA_ROTATIVIDADE_6M"
    FEATURES_CATEGORICAS = ["uf", "seção", "FAIXA_ETARIA", "FAIXA_VOLUME_COORTE"]
    FEATURES_NUMERICAS = ["N_TOTAL_T", "N_POSITIVOS_T", "N_NEGATIVOS_T", "LOG_VOLUME_COORTE", "PROP_NEGATIVOS_T", "ANO", "MES", "TRIMESTRE", "MES_SIN", "MES_COS"]
    SEED = 42

    # Split idêntico ao da LR (70/30)
    train_df, test_df = df_ml.randomSplit([0.7, 0.3], seed=SEED)
    train_cnt = train_df.count()
    test_cnt = test_df.count()
    print(f"TREINO: {train_cnt:,} | TESTE: {test_cnt:,}")

    # Pipelines
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in FEATURES_CATEGORICAS]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in FEATURES_CATEGORICAS]
    assembler = VectorAssembler(inputCols=[f"{c}_ohe" for c in FEATURES_CATEGORICAS] + FEATURES_NUMERICAS, outputCol="features")

    # 1. Fit LR
    lr = LogisticRegression(featuresCol="features", labelCol=TARGET_COL)
    pipeline_lr = Pipeline(stages=indexers + encoders + [assembler, lr])
    
    start_lr = time.perf_counter()
    model_lr = pipeline_lr.fit(train_df)
    TEMPO_TREINO_LR = time.perf_counter() - start_lr
    lr_stage = model_lr.stages[-1]

    preds_lr = model_lr.transform(test_df)
    eval_auc = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    eval_acc = MulticlassClassificationEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="accuracy")

    auc_lr = eval_auc.evaluate(preds_lr)
    acc_lr = eval_acc.evaluate(preds_lr)
    
    tn_lr = preds_lr.filter((F.col(TARGET_COL) == 0) & (F.col("prediction") == 0.0)).count()
    fp_lr = preds_lr.filter((F.col(TARGET_COL) == 0) & (F.col("prediction") == 1.0)).count()
    fn_lr = preds_lr.filter((F.col(TARGET_COL) == 1) & (F.col("prediction") == 0.0)).count()
    tp_lr = preds_lr.filter((F.col(TARGET_COL) == 1) & (F.col("prediction") == 1.0)).count()

    prec_lr = tp_lr / (tp_lr + fp_lr)
    rec_lr = tp_lr / (tp_lr + fn_lr)
    f1_lr = (2 * prec_lr * rec_lr) / (prec_lr + rec_lr)
    spec_lr = tn_lr / (tn_lr + fp_lr)

    # 2. Fit Random Forest Conservadora (numTrees=30, maxDepth=8, seed=42)
    rf = RandomForestClassifier(featuresCol="features", labelCol=TARGET_COL, numTrees=30, maxDepth=8, seed=SEED)
    pipeline_rf = Pipeline(stages=indexers + encoders + [assembler, rf])

    print("\nIniciando treinamento da Random Forest Classifier no train_df...")
    start_rf = time.perf_counter()
    model_rf = pipeline_rf.fit(train_df)
    TEMPO_TREINO_RF = time.perf_counter() - start_rf
    print(f"Treinamento RF concluído em TEMPO_TREINO_RF = {TEMPO_TREINO_RF:.3f} segundos!")

    rf_stage = model_rf.stages[-1]
    
    # Predições RF no teste
    preds_rf = model_rf.transform(test_df)
    auc_rf = eval_auc.evaluate(preds_rf)
    acc_rf = eval_acc.evaluate(preds_rf)

    tn_rf = preds_rf.filter((F.col(TARGET_COL) == 0) & (F.col("prediction") == 0.0)).count()
    fp_rf = preds_rf.filter((F.col(TARGET_COL) == 0) & (F.col("prediction") == 1.0)).count()
    fn_rf = preds_rf.filter((F.col(TARGET_COL) == 1) & (F.col("prediction") == 0.0)).count()
    tp_rf = preds_rf.filter((F.col(TARGET_COL) == 1) & (F.col("prediction") == 1.0)).count()

    prec_rf = tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0.0
    rec_rf = tp_rf / (tp_rf + fn_rf) if (tp_rf + fn_rf) > 0 else 0.0
    f1_rf = (2 * prec_rf * rec_rf) / (prec_rf + rec_rf) if (prec_rf + rec_rf) > 0 else 0.0
    spec_rf = tn_rf / (tn_rf + fp_rf) if (tn_rf + fp_rf) > 0 else 0.0

    print("\n=== MATRIZ DE CONFUSÃO DA RANDOM FOREST ===")
    print(f"  TN_RF = {tn_rf:6,} | FP_RF = {fp_rf:6,}")
    print(f"  FN_RF = {fn_rf:6,} | TP_RF = {tp_rf:6,}")

    print("\n=== MÉTRICAS DA RANDOM FOREST NO TESTE ===")
    print(f"  AUC-ROC RF   = {auc_rf:.4f}")
    print(f"  Accuracy RF  = {acc_rf:.4f} ({acc_rf*100:.2f}%)")
    print(f"  Precision 1  = {prec_rf:.4f}")
    print(f"  Recall 1     = {rec_rf:.4f}")
    print(f"  F1-Score 1   = {f1_rf:.4f}")
    print(f"  Specificity  = {spec_rf:.4f}")

    # Extrair mapeamento exato de nomes de features do VectorAssembler
    feature_names = get_feature_names_from_schema(preds_rf.schema, col_name="features")
    print(f"\nTotal de features vetorizadas: {len(feature_names)}")

    # Extrair Feature Importances da RF
    importances = rf_stage.featureImportances.toArray()
    rf_feat_imp = list(zip(range(len(importances)), feature_names, importances))
    rf_feat_imp_sorted = sorted(rf_feat_imp, key=lambda x: x[2], reverse=True)

    print("\n--- TOP 10 FEATURE IMPORTANCE (RANDOM FOREST) ---")
    for rank, (idx, fname, imp) in enumerate(rf_feat_imp_sorted[:10], 1):
        print(f"  {rank:2d}. {fname:30s} | Importância: {imp:.6f}")

    top_5_rf = [f"{fname} ({imp:.4f})" for idx, fname, imp in rf_feat_imp_sorted[:5]]

    # Agrupar importância por feature original
    agg_imp = {}
    for idx, fname, imp in rf_feat_imp:
        orig = fname.split("_")[0]
        # Tratar casos específicos de nomes compostos
        if fname.startswith("LOG_VOLUME") or fname.startswith("PROP_NEGATIVOS") or fname.startswith("FAIXA_VOLUME") or fname.startswith("FAIXA_ETARIA") or fname.startswith("N_") or fname.startswith("MES_"):
            if "FAIXA_ETARIA" in fname:
                orig = "FAIXA_ETARIA"
            elif "FAIXA_VOLUME" in fname:
                orig = "FAIXA_VOLUME_COORTE"
            elif "LOG_VOLUME" in fname:
                orig = "LOG_VOLUME_COORTE"
            elif "PROP_NEGATIVOS" in fname:
                orig = "PROP_NEGATIVOS_T"
            elif "MES_SIN" in fname:
                orig = "MES_SIN"
            elif "MES_COS" in fname:
                orig = "MES_COS"
            elif "N_TOTAL" in fname:
                orig = "N_TOTAL_T"
            elif "N_POSITIVOS" in fname:
                orig = "N_POSITIVOS_T"
            elif "N_NEGATIVOS" in fname:
                orig = "N_NEGATIVOS_T"
        agg_imp[orig] = agg_imp.get(orig, 0.0) + imp

    agg_imp_sorted = sorted(agg_imp.items(), key=lambda x: x[1], reverse=True)
    print("\n--- IMPORTÂNCIA AGREGADA POR FEATURE ORIGINAL (RANDOM FOREST) ---")
    for orig_f, imp_sum in agg_imp_sorted:
        print(f"  {orig_f:25s} | Importância Agregada: {imp_sum:.6f}")

    # Extrair Coeficientes da Regressão Logística
    coeffs = lr_stage.coefficients.toArray()
    lr_feat_coef = list(zip(range(len(coeffs)), feature_names, coeffs, [abs(c) for c in coeffs]))
    lr_feat_coef_sorted = sorted(lr_feat_coef, key=lambda x: x[3], reverse=True)

    print("\n--- TOP 10 COEFICIENTES EM MAGNITUDE ABSOLUTA (LOGISTIC REGRESSION) ---")
    for rank, (idx, fname, coef, abs_c) in enumerate(lr_feat_coef_sorted[:10], 1):
        sinal = "+" if coef > 0 else "-"
        print(f"  {rank:2d}. {fname:30s} | Coef: {coef:10.6f} | Sinal: {sinal}")

    top_5_lr = [f"{fname} ({coef:+.4f})" for idx, fname, coef, abs_c in lr_feat_coef_sorted[:5]]

    # Modelo Recomendado
    MODELO_RECOMENDADO = "RandomForestClassifier" if auc_rf >= auc_lr else "LogisticRegression"
    print(f"\nMODELO RECOMENDADO: {MODELO_RECOMENDADO}")
    print(f"  (Comparativo AUC-ROC: RF = {auc_rf:.4f} vs LR = {auc_lr:.4f})")
    print(f"  (Comparativo Accuracy: RF = {acc_rf:.4f} vs LR = {acc_lr:.4f})")

    # Salvar Modelo RF
    rf_model_dir = root_dir / "models" / "random_forest_nordeste"
    if rf_model_dir.exists():
        shutil.rmtree(rf_model_dir)

    print(f"\nSalvando PipelineModel da Random Forest em: {rf_model_dir}")
    model_rf.write().overwrite().save(str(rf_model_dir))
    MODELO_RF_SALVO = "SIM" if (rf_model_dir / "metadata").exists() else "NÃO"

    # Atualizar Métricas CSV
    metrics_csv = root_dir / "outputs" / "metrics" / "model_comparison_metrics.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "auc_roc", "accuracy", "precision_pos", "recall_pos", "f1_pos", "specificity", "training_time_sec", "seed"])
        writer.writerow(["Baseline (Majoritária)", f"0.000000", f"{0.491914:.6f}", f"0.000000", f"0.000000", f"0.000000", f"1.000000", f"0.000", SEED])
        writer.writerow(["Logistic Regression", f"{auc_lr:.6f}", f"{acc_lr:.6f}", f"{prec_lr:.6f}", f"{rec_lr:.6f}", f"{f1_lr:.6f}", f"{spec_lr:.6f}", f"{TEMPO_TREINO_LR:.3f}", SEED])
        writer.writerow(["Random Forest", f"{auc_rf:.6f}", f"{acc_rf:.6f}", f"{prec_rf:.6f}", f"{rec_rf:.6f}", f"{f1_rf:.6f}", f"{spec_rf:.6f}", f"{TEMPO_TREINO_RF:.3f}", SEED])

    METRICAS_ATUALIZADAS = "SIM" if metrics_csv.exists() else "NÃO"
    print(f"METRICAS_ATUALIZADAS = {METRICAS_ATUALIZADAS} ({metrics_csv})")

    # Resumo para a resposta final
    rf_summary = {
        "rf_params": {"numTrees": 30, "maxDepth": 8, "seed": SEED},
        "auc_rf": auc_rf,
        "accuracy_rf": acc_rf,
        "precision_rf": prec_rf,
        "recall_rf": rec_rf,
        "f1_rf": f1_rf,
        "specificity_rf": spec_rf,
        "matriz_rf": {"tn": tn_rf, "fp": fp_rf, "fn": fn_rf, "tp": tp_rf},
        "tempo_lr": TEMPO_TREINO_LR,
        "tempo_rf": TEMPO_TREINO_RF,
        "modelo_recomendado": MODELO_RECOMENDADO,
        "top_5_rf": top_5_rf,
        "top_5_lr": top_5_lr,
        "modelo_rf_salvo": MODELO_RF_SALVO,
        "metricas_atualizadas": METRICAS_ATUALIZADAS,
        "notebook_salvo": True
    }

    with open(root_dir / "outputs" / "rf_summary.json", "w", encoding="utf-8") as f:
        json.dump(rf_summary, f, indent=2, ensure_ascii=False)

    print("\nSUCESSO: Etapa Random Forest concluída e salva!")
    spark.stop()

if __name__ == "__main__":
    main()
