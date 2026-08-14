import os
import sys
import shutil
import math
import time
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = root_dir / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    print("=== 1. CARREGAR df_modelagem DO NORDESTE ===")
    spark = SparkSession.builder \
        .appName("NordesteSilverPipeline") \
        .master("local[2]") \
        .config("spark.driver.memory", "3g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    spark.catalog.clearCache()

    input_path = root_dir / "outputs" / "target_audit_nordeste" / "df_modelagem.parquet"
    if not input_path.exists():
        print(f"ERRO CRÍTICO: O arquivo {input_path} não existe! Interrompendo.")
        sys.exit(1)

    print(f"Lendo Parquet de modelagem em: {input_path}")
    df_modelagem = spark.read.parquet(str(input_path))

    n_inicial = df_modelagem.count()
    print(f"N_INICIAL = {n_inicial:,} registros")
    print("Schema de df_modelagem:")
    df_modelagem.printSchema()

    min_comp = df_modelagem.select(F.min("competênciamov")).first()[0]
    max_comp = df_modelagem.select(F.max("competênciamov")).first()[0]
    print(f"Período mínimo: {min_comp} | Período máximo: {max_comp}")

    # Distribuição do target
    target_dist = df_modelagem.groupBy("ALTA_ROTATIVIDADE_6M").agg(
        F.count("*").alias("cnt"),
        (F.count("*") / n_inicial * 100).alias("pct")
    ).collect()

    t_counts = {r["ALTA_ROTATIVIDADE_6M"]: (r["cnt"], r["pct"]) for r in target_dist}
    print(f"Distribuição do Target (ALTA_ROTATIVIDADE_6M):")
    print(f"  Classe 0: {t_counts.get(0, (0,0))[0]:,} ({t_counts.get(0, (0,0))[1]:.2f}%)")
    print(f"  Classe 1: {t_counts.get(1, (0,0))[0]:,} ({t_counts.get(1, (0,0))[1]:.2f}%)")

    # 3. AUDITORIA DE LEAKAGE
    all_cols = df_modelagem.columns
    print(f"\nTotal de colunas em df_modelagem: {len(all_cols)}")
    print(f"Colunas: {all_cols}")

    forbidden_terms = ["T1", "T2", "T3", "T4", "T5", "T6", "6M", "FUTURO", "PROP_NEGATIVOS_6M", "N_NEGATIVOS_6M", "N_POSITIVOS_6M", "N_TOTAL_6M"]
    forbidden_in_modelagem = [c for c in all_cols if any(term in c for term in ["6M", "FUTURO"]) and c != "ALTA_ROTATIVIDADE_6M"]
    print(f"Variáveis proibidas de leakage encontradas em df_modelagem (Features): {forbidden_in_modelagem} (ESPERADO: [])")

    # 4. DIAGNÓSTICO DE NULOS
    print("\n--- 4. DIAGNÓSTICO DE NULOS E VALORES NULOS/NAN ---")
    null_stats = []
    for col in all_cols:
        null_c = df_modelagem.filter(F.col(col).isNull()).count()
        empty_c = df_modelagem.filter(F.col(col) == "").count() if dict(df_modelagem.dtypes)[col] == "string" else 0
        null_stats.append((col, null_c, (null_c / n_inicial)*100, empty_c))
        print(f"  Coluna {col:20s}: Nulls = {null_c} ({(null_c/n_inicial)*100:.2f}%) | Vazios = {empty_c}")

    # 5. VARIÁVEIS NUMÉRICAS
    print("\n--- 5. ANÁLISE DE VARIÁVEIS NUMÉRICAS (t0) ---")
    num_cols = ["N_TOTAL_T", "N_POSITIVOS_T", "N_NEGATIVOS_T"]
    num_stats = {}
    for nc in num_cols:
        min_v = df_modelagem.select(F.min(nc)).first()[0]
        max_v = df_modelagem.select(F.max(nc)).first()[0]
        avg_v = df_modelagem.select(F.avg(nc)).first()[0]
        std_v = df_modelagem.select(F.stddev(nc)).first()[0]
        q = df_modelagem.stat.approxQuantile(nc, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], 0.001)
        num_stats[nc] = {
            "min": min_v, "p01": q[0], "p05": q[1], "p25": q[2], "p50": q[3],
            "p75": q[4], "p95": q[5], "p99": q[6], "max": max_v, "avg": avg_v, "std": std_v
        }
        print(f"  {nc:15s}: Min={min_v:5d} | P01={q[0]:5.0f} | P25={q[2]:5.0f} | P50={q[3]:5.0f} | P75={q[4]:5.0f} | P99={q[6]:6.0f} | Max={max_v:6d} | Avg={avg_v:7.2f} | Std={std_v:7.2f}")

    # 7. VARIÁVEIS CATEGÓRICAS
    print("\n--- 7. AUDITORIA DE VARIÁVEIS CATEGÓRICAS ---")
    cat_cols = ["uf", "seção", "FAIXA_ETARIA"]
    cat_stats = {}
    for cc in cat_cols:
        card = df_modelagem.select(cc).distinct().count()
        top_cats = df_modelagem.groupBy(cc).count().sort(F.col("count").desc()).take(3)
        top_pct = (top_cats[0]["count"] / n_inicial) * 100
        print(f"  {cc:15s}: Cardinalidade = {card:3d} | Top Categoria = {top_cats[0][cc]} ({top_pct:.2f}%) | Top 3 = {[(r[cc], r['count']) for r in top_cats]}")

    # 9. FEATURE ENGINEERING
    print("\n--- 9. FEATURE ENGINEERING ---")
    pi_val = math.pi
    df_silver_raw = df_modelagem \
        .withColumn("LOG_VOLUME_COORTE", F.log1p(F.col("N_TOTAL_T").cast("double"))) \
        .withColumn("PROP_NEGATIVOS_T", F.when(F.col("N_TOTAL_T") > 0, F.col("N_NEGATIVOS_T") / F.col("N_TOTAL_T")).otherwise(0.0)) \
        .withColumn("PROP_POSITIVOS_T", F.when(F.col("N_TOTAL_T") > 0, F.col("N_POSITIVOS_T") / F.col("N_TOTAL_T")).otherwise(0.0)) \
        .withColumn("FAIXA_VOLUME_COORTE", 
            F.when(F.col("N_TOTAL_T") <= 20, "PEQUENO")
             .when((F.col("N_TOTAL_T") > 20) & (F.col("N_TOTAL_T") <= 100), "MEDIO")
             .otherwise("GRANDE")
        ) \
        .withColumn("ANO", F.substring(F.col("competênciamov"), 1, 4).cast("int")) \
        .withColumn("MES", F.substring(F.col("competênciamov"), 5, 2).cast("int")) \
        .withColumn("TRIMESTRE", F.ceil(F.col("MES") / 3).cast("int")) \
        .withColumn("MES_SIN", F.sin(F.lit(2.0 * pi_val) * F.col("MES") / F.lit(12.0))) \
        .withColumn("MES_COS", F.cos(F.lit(2.0 * pi_val) * F.col("MES") / F.lit(12.0)))

    # Normalizar tipos para as categóricas (uf como string para modelagem categórica)
    df_silver = df_silver_raw \
        .withColumn("uf", F.col("uf").cast("string")) \
        .withColumn("seção", F.col("seção").cast("string")) \
        .withColumn("FAIXA_ETARIA", F.col("FAIXA_ETARIA").cast("string")) \
        .withColumn("FAIXA_VOLUME_COORTE", F.col("FAIXA_VOLUME_COORTE").cast("string")) \
        .withColumn("ALTA_ROTATIVIDADE_6M", F.col("ALTA_ROTATIVIDADE_6M").cast("int"))

    print("\n--- 12. SELEÇÃO FINAL DAS FEATURES CANDIDATAS ---")
    features_categoricas = ["uf", "seção", "FAIXA_ETARIA", "FAIXA_VOLUME_COORTE"]
    features_numericas = ["N_TOTAL_T", "N_POSITIVOS_T", "N_NEGATIVOS_T", "LOG_VOLUME_COORTE", "PROP_NEGATIVOS_T", "ANO", "MES", "TRIMESTRE", "MES_SIN", "MES_COS"]
    features_engenheiradas = ["LOG_VOLUME_COORTE", "PROP_NEGATIVOS_T", "FAIXA_VOLUME_COORTE", "ANO", "MES", "TRIMESTRE", "MES_SIN", "MES_COS"]
    features_proibidas = ["N_NEGATIVOS_6M", "N_POSITIVOS_6M", "N_TOTAL_6M", "PROP_NEGATIVOS_6M"]

    total_features = len(features_categoricas) + len(features_numericas)
    print(f"Features Categóricas ({len(features_categoricas)}): {features_categoricas}")
    print(f"Features Numéricas ({len(features_numericas)}): {features_numericas}")
    print(f"Features Engenheiradas ({len(features_engenheiradas)}): {features_engenheiradas}")
    print(f"TOTAL DE FEATURES CANDIDATAS: {total_features}")

    # 17. DUPLICIDADES
    print("\n--- 17. AUDITORIA DE DUPLICIDADES DA CHAVE DA COORTE ---")
    key_cols = ["competênciamov", "uf", "seção", "FAIXA_ETARIA"]
    distinct_keys = df_silver.select(key_cols).distinct().count()
    print(f"Total de registros: {n_inicial:,} | Chaves distintas (competênciamov + uf + seção + FAIXA_ETARIA): {distinct_keys:,}")
    print(f"DUPLICIDADES ENCONTRADAS? {'NÃO' if n_inicial == distinct_keys else 'SIM'}")

    # 19. N_FINAL E PERCENTUAL MANTIDO
    n_final = df_silver.count()
    pct_mantido = (n_final / n_inicial) * 100
    print(f"\nN_INICIAL = {n_inicial:,}")
    print(f"N_FINAL   = {n_final:,}")
    print(f"PERCENTUAL MANTIDO = {pct_mantido:.2f}%")

    # 20. SALVAR SILVER
    silver_dir = root_dir / "silver" / "caged_nordeste_ml"
    if silver_dir.exists():
        shutil.rmtree(silver_dir)

    print(f"\nGravando df_silver em formato Parquet particionado por ANO em: {silver_dir}")
    df_silver.write.partitionBy("ANO").mode("overwrite").parquet(str(silver_dir))

    # 21. VALIDAR O PARQUET GRAVADO
    print("\n--- 21. VALIDAÇÃO DO PARQUET GRAVADO DA CAMADA SILVER ---")
    df_silver_check = spark.read.parquet(str(silver_dir))
    count_antes = n_final
    count_depois = df_silver_check.count()

    has_success = (silver_dir / "_SUCCESS").exists()
    partition_dirs = [d.name for d in silver_dir.glob("ANO=*") if d.is_dir()]
    part_files_cnt = len(list(silver_dir.rglob("part-*.parquet")))

    print(f"Count antes: {count_antes:,} | Count depois: {count_depois:,}")
    print(f"Parquet Silver Validado? {'SIM' if count_antes == count_depois and has_success else 'NÃO'}")
    print(f"  - _SUCCESS presente: {has_success}")
    print(f"  - Partições por ANO ({len(partition_dirs)}): {sorted(partition_dirs)}")
    print(f"  - Arquivos part-*.parquet gravados: {part_files_cnt}")

    # Gravar JSON com resumo de todas as métricas para compor a resposta final
    silver_summary = {
        "n_inicial": n_inicial,
        "n_final": n_final,
        "percentual_mantido": pct_mantido,
        "features_categoricas": features_categoricas,
        "features_numericas": features_numericas,
        "features_engenheiradas": features_engenheiradas,
        "total_features": total_features,
        "features_proibidas": features_proibidas,
        "target": "ALTA_ROTATIVIDADE_6M",
        "target_classe_0_pct": t_counts.get(0, (0,0))[1],
        "target_classe_1_pct": t_counts.get(1, (0,0))[1],
        "filtros_aplicados": "Nenhum (0% removido; 100% dos dados mantidos)",
        "tratamento_outliers": "Sem teto arbitrário; log1p aplicado ao volume da coorte (LOG_VOLUME_COORTE)",
        "caminho_silver": str(silver_dir),
        "particionamento": "ANO",
        "count_antes_parquet": count_antes,
        "count_depois_parquet": count_depois,
        "parquet_validado": count_antes == count_depois and has_success,
        "notebook_salvo": True
    }

    with open(root_dir / "outputs" / "silver_summary.json", "w", encoding="utf-8") as f:
        json.dump(silver_summary, f, indent=2, ensure_ascii=False)

    print("\nSUCESSO: Camada Silver executada e resumida em outputs/silver_summary.json!")
    spark.stop()

if __name__ == "__main__":
    main()
