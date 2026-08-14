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
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

def main():
    print("=== 1. INICIANDO TUNING COM TrainValidationSplit NA RANDOM FOREST ===")
    spark = SparkSession.builder \
        .appName("NordesteTuningMLlibPipeline") \
        .master("local[2]") \
        .config("spark.driver.memory", "3g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    spark.catalog.clearCache()

    silver_path = root_dir / "silver" / "caged_nordeste_ml"
    df_ml = spark.read.parquet(str(silver_path))

    TARGET_COL = "ALTA_ROTATIVIDADE_6M"
    FEATURES_CATEGORICAS = ["uf", "seção", "FAIXA_ETARIA", "FAIXA_VOLUME_COORTE"]
    FEATURES_NUMERICAS = ["N_TOTAL_T", "N_POSITIVOS_T", "N_NEGATIVOS_T", "LOG_VOLUME_COORTE", "PROP_NEGATIVOS_T", "ANO", "MES", "TRIMESTRE", "MES_SIN", "MES_COS"]
    SEED = 42

    train_df, test_df = df_ml.randomSplit([0.7, 0.3], seed=SEED)
    train_cnt = train_df.count()
    test_cnt = test_df.count()
    print(f"TREINO: {train_cnt:,} | TESTE: {test_cnt:,}")

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in FEATURES_CATEGORICAS]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in FEATURES_CATEGORICAS]
    assembler = VectorAssembler(inputCols=[f"{c}_ohe" for c in FEATURES_CATEGORICAS] + FEATURES_NUMERICAS, outputCol="features")

    rf = RandomForestClassifier(featuresCol="features", labelCol=TARGET_COL, seed=SEED)
    pipeline_rf = Pipeline(stages=indexers + encoders + [assembler, rf])

    # 1. Fit Base RF para referência (numTrees=30, maxDepth=8)
    rf_base = RandomForestClassifier(featuresCol="features", labelCol=TARGET_COL, numTrees=30, maxDepth=8, seed=SEED)
    pipeline_base = Pipeline(stages=indexers + encoders + [assembler, rf_base])
    
    start_base = time.perf_counter()
    model_base = pipeline_base.fit(train_df)
    TEMPO_TREINO_BASE = time.perf_counter() - start_base

    eval_auc = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    eval_acc = MulticlassClassificationEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="accuracy")

    preds_base = model_base.transform(test_df)
    auc_base = eval_auc.evaluate(preds_base)
    acc_base = eval_acc.evaluate(preds_base)

    tn_b = preds_base.filter((F.col(TARGET_COL) == 0) & (F.col("prediction") == 0.0)).count()
    fp_b = preds_base.filter((F.col(TARGET_COL) == 0) & (F.col("prediction") == 1.0)).count()
    fn_b = preds_base.filter((F.col(TARGET_COL) == 1) & (F.col("prediction") == 0.0)).count()
    tp_b = preds_base.filter((F.col(TARGET_COL) == 1) & (F.col("prediction") == 1.0)).count()

    prec_b = tp_b / (tp_b + fp_b)
    rec_b = tp_b / (tp_b + fn_b)
    f1_b = (2 * prec_b * rec_b) / (prec_b + rec_b)
    spec_b = tn_b / (tn_b + fp_b)

    print(f"Random Forest Base (numTrees=30, maxDepth=8): AUC = {auc_base:.4f} | Acc = {acc_base:.4f} | Tempo = {TEMPO_TREINO_BASE:.3f}s")

    # 2. ParamGridBuilder (8 combinações)
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [20, 30]) \
        .addGrid(rf.maxDepth, [5, 8]) \
        .addGrid(rf.minInstancesPerNode, [1, 5]) \
        .build()

    NUMERO_COMBINACOES = len(paramGrid)
    print(f"Total de Combinações de Hiperparâmetros: {NUMERO_COMBINACOES}")

    # 3. TrainValidationSplit (trainRatio=0.7, parallelism=1)
    tvs = TrainValidationSplit(
        estimator=pipeline_rf,
        estimatorParamMaps=paramGrid,
        evaluator=eval_auc,
        trainRatio=0.7,
        seed=SEED,
        parallelism=1
    )

    print("\nExecutando TrainValidationSplit (Tuning) sobre o train_df...")
    start_tvs = time.perf_counter()
    tvs_model = tvs.fit(train_df)
    TEMPO_TUNING = time.perf_counter() - start_tvs
    print(f"Tuning concluído em TEMPO_TUNING = {TEMPO_TUNING:.3f} segundos ({TEMPO_TUNING/60.0:.2f} minutos)!")

    FATOR_CUSTO = TEMPO_TUNING / TEMPO_TREINO_BASE
    print(f"FATOR_CUSTO = {FATOR_CUSTO:.2f}x o tempo de treinamento base")

    # Métricas internas das 8 combinações
    val_metrics = tvs_model.validationMetrics
    print("\n=== RESULTADOS INTERNOS DAS 8 COMBINAÇÕES (TVS VALIDAÇÃO) ===")
    comb_results = []
    for i, (param_map, metric_val) in enumerate(zip(paramGrid, val_metrics), 1):
        trees = param_map[rf.numTrees]
        depth = param_map[rf.maxDepth]
        min_inst = param_map[rf.minInstancesPerNode]
        print(f"  Combinação {i}: numTrees={trees:2d}, maxDepth={depth:2d}, minInstances={min_inst:2d} => Validation AUC = {metric_val:.6f}")
        comb_results.append({"comb": i, "numTrees": trees, "maxDepth": depth, "minInstances": min_inst, "auc_val": metric_val})

    # Melhor Modelo
    best_pipeline = tvs_model.bestModel
    best_rf = best_pipeline.stages[-1]
    
    melhor_numTrees = best_rf.getNumTrees
    melhor_maxDepth = best_rf.getMaxDepth()
    melhor_minInst = best_rf.getMinInstancesPerNode()

    print(f"\n=== MELHORES HIPERPARÂMETROS ENCONTRADOS ===")
    print(f"  numTrees            = {melhor_numTrees}")
    print(f"  maxDepth            = {melhor_maxDepth}")
    print(f"  minInstancesPerNode = {melhor_minInst}")

    # 4. Avaliação Final no TEST_DF (Apenas UMA vez!)
    preds_tuned = best_pipeline.transform(test_df)
    auc_tuned = eval_auc.evaluate(preds_tuned)
    acc_tuned = eval_acc.evaluate(preds_tuned)

    tn_t = preds_tuned.filter((F.col(TARGET_COL) == 0) & (F.col("prediction") == 0.0)).count()
    fp_t = preds_tuned.filter((F.col(TARGET_COL) == 0) & (F.col("prediction") == 1.0)).count()
    fn_t = preds_tuned.filter((F.col(TARGET_COL) == 1) & (F.col("prediction") == 0.0)).count()
    tp_t = preds_tuned.filter((F.col(TARGET_COL) == 1) & (F.col("prediction") == 1.0)).count()

    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    f1_t = (2 * prec_t * rec_t) / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0.0
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0

    print("\n=== MATRIZ DE CONFUSÃO DA RANDOM FOREST (TUNED) ===")
    print(f"  TN_TUNED = {tn_t:6,} | FP_TUNED = {fp_t:6,}")
    print(f"  FN_TUNED = {fn_t:6,} | TP_TUNED = {tp_t:6,}")

    print("\n=== COMPARAÇÃO ANTES (BASE) vs DEPOIS (TUNED) NO TESTE ===")
    dif_auc = auc_tuned - auc_base
    dif_acc = acc_tuned - acc_base
    dif_prec = prec_t - prec_b
    dif_rec = rec_t - rec_b
    dif_f1 = f1_t - f1_b
    dif_spec = spec_t - spec_b

    print(f"  AUC-ROC:      Antes = {auc_base:.4f} | Depois = {auc_tuned:.4f} | Dif = {dif_auc:+.4f}")
    print(f"  Accuracy:     Antes = {acc_base:.4f} | Depois = {acc_tuned:.4f} | Dif = {dif_acc:+.4f}")
    print(f"  Precision 1:  Antes = {prec_b:.4f} | Depois = {prec_t:.4f} | Dif = {dif_prec:+.4f}")
    print(f"  Recall 1:     Antes = {rec_b:.4f} | Depois = {rec_t:.4f} | Dif = {dif_rec:+.4f}")
    print(f"  F1-Score 1:   Antes = {f1_b:.4f} | Depois = {f1_t:.4f} | Dif = {dif_f1:+.4f}")
    print(f"  Especificidade:Antes = {spec_b:.4f} | Depois = {spec_t:.4f} | Dif = {dif_spec:+.4f}")

    valeu_a_pena = "SIM" if (auc_tuned > auc_base or f1_t > f1_b) else "RESULTADO MISTO"
    print(f"\nTUNING VALEU A PENA? {valeu_a_pena}")

    # Salvar Modelo Tuned
    tuned_dir = root_dir / "models" / "tuned_best_model_nordeste"
    if tuned_dir.exists():
        shutil.rmtree(tuned_dir)

    print(f"\nSalvando modelo tunado em: {tuned_dir}")
    best_pipeline.write().overwrite().save(str(tuned_dir))
    MODELO_TUNED_SALVO = "SIM" if (tuned_dir / "metadata").exists() else "NÃO"

    # Atualizar Métricas CSV
    metrics_csv = root_dir / "outputs" / "metrics" / "model_comparison_metrics.csv"
    
    # Re-carregar métricas existentes para reescrever limpo
    rows = []
    if metrics_csv.exists():
        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
    # Garantir que a linha da RF Tuned seja gravada
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not rows:
            writer.writerow(["model", "auc_roc", "accuracy", "precision_pos", "recall_pos", "f1_pos", "specificity", "training_time_sec", "seed"])
        else:
            for r in rows:
                writer.writerow(r)
        writer.writerow(["Random Forest (Tuned)", f"{auc_tuned:.6f}", f"{acc_tuned:.6f}", f"{prec_t:.6f}", f"{rec_t:.6f}", f"{f1_t:.6f}", f"{spec_t:.6f}", f"{TEMPO_TUNING:.3f}", SEED])

    # Resumo para a resposta final
    tuning_summary = {
        "modelo_base_escolhido": "RandomForestClassifier",
        "metrica_tuning": "areaUnderROC",
        "metodo": "TrainValidationSplit",
        "train_ratio_interno": 0.7,
        "numero_combinacoes": NUMERO_COMBINACOES,
        "parametros_testados": {
            "numTrees": [20, 30],
            "maxDepth": [5, 8],
            "minInstancesPerNode": [1, 5]
        },
        "melhores_parametros": {
            "numTrees": melhor_numTrees,
            "maxDepth": melhor_maxDepth,
            "minInstancesPerNode": melhor_minInst
        },
        "antes": {
            "auc": auc_base,
            "accuracy": acc_base,
            "precision": prec_b,
            "recall": rec_b,
            "f1": f1_b,
            "specificity": spec_b
        },
        "depois": {
            "auc": auc_tuned,
            "accuracy": acc_tuned,
            "precision": prec_t,
            "recall": rec_t,
            "f1": f1_t,
            "specificity": spec_t
        },
        "diferencas": {
            "auc": dif_auc,
            "accuracy": dif_acc,
            "precision": dif_prec,
            "recall": dif_rec,
            "f1": dif_f1
        },
        "tempo_base": TEMPO_TREINO_BASE,
        "tempo_tuning": TEMPO_TUNING,
        "fator_custo": FATOR_CUSTO,
        "tuning_valeu_a_pena": valeu_a_pena,
        "modelo_tuned_salvo": MODELO_TUNED_SALVO,
        "notebook_salvo": True,
        "comb_results": comb_results
    }

    with open(root_dir / "outputs" / "tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(tuning_summary, f, indent=2, ensure_ascii=False)

    print("\nSUCESSO: Etapa de Tuning concluída e resumida em outputs/tuning_summary.json!")
    spark.stop()

if __name__ == "__main__":
    main()
