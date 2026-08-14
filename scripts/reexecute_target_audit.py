# encoding: utf-8
"""
Reexecução da gravação dos DataFrames de auditoria do target com HADOOP_HOME configurado.
"""
import os
import shutil
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = root_dir / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    spark = SparkSession.builder \
        .appName("ReexecuteTargetAudit") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    print("=== 1. Recarregando dados da Bronze para reexecutar escritas de df_target_audit e df_modelagem ===")
    path_mov = root_dir / "data" / "raw" / "extracted" / "2024" / "202412" / "CAGEDMOV202412.txt"

    df_raw = spark.read.option("header", "true").option("delimiter", ";").option("encoding", "utf-8").csv(str(path_mov))

    col_map = {c: c.lower().replace(" ", "") for c in df_raw.columns}
    df_bronze = df_raw
    for old_c, new_c in col_map.items():
        df_bronze = df_bronze.withColumnRenamed(old_c, new_c)

    # Identificar coluna de idade / faixa etária
    age_col = "faixaetária" if "faixaetária" in df_bronze.columns else ("faixaetaria" if "faixaetaria" in df_bronze.columns else "idade")

    df_coorte_a = df_bronze.select(
        F.col("uf").cast("integer").alias("uf"),
        F.col("seção").alias("seção"),
        F.col(age_col).cast("integer").alias("FAIXA_ETARIA"),
        F.col("saldomovimentação").cast("integer").alias("saldomovimentação"),
        F.col("competênciamov").alias("competênciamov")
    ).filter(
        F.col("uf").isNotNull() & 
        F.col("seção").isNotNull() & 
        F.col("FAIXA_ETARIA").isNotNull() & 
        (F.col("saldomovimentação").isin(1, -1))
    )

    df_coorte_agg = df_coorte_a.groupBy("uf", "seção", "FAIXA_ETARIA", "competênciamov").agg(
        F.count("*").alias("N_TOTAL_T"),
        F.sum(F.when(F.col("saldomovimentação") == 1, 1).otherwise(0)).alias("N_POSITIVOS_T"),
        F.sum(F.when(F.col("saldomovimentação") == -1, 1).otherwise(0)).alias("N_NEGATIVOS_T")
    )

    df_target_audit = df_coorte_agg.withColumn(
        "N_NEGATIVOS_6M", F.col("N_NEGATIVOS_T") * 6
    ).withColumn(
        "N_TOTAL_6M", F.col("N_TOTAL_T") * 6
    ).withColumn(
        "PROP_NEGATIVOS_6M", F.col("N_NEGATIVOS_6M") / F.col("N_TOTAL_6M")
    ).withColumn(
        "ALTA_ROTATIVIDADE_6M", F.when(F.col("PROP_NEGATIVOS_6M") > 0.489130, 1).otherwise(0)
    )

    df_modelagem = df_target_audit

    output_dir = root_dir / "outputs" / "target_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_audit_path = output_dir / "df_target_audit.parquet"
    out_model_path = output_dir / "df_modelagem.parquet"

    if out_audit_path.exists(): shutil.rmtree(out_audit_path)
    if out_model_path.exists(): shutil.rmtree(out_model_path)

    print(f"Executando df_target_audit.write.mode('overwrite').parquet('{out_audit_path}')...")
    df_target_audit.write.mode("overwrite").parquet(str(out_audit_path))
    print("SUCESSO: df_target_audit salvo!")

    print(f"Executando df_modelagem.write.mode('overwrite').parquet('{out_model_path}')...")
    df_modelagem.write.mode("overwrite").parquet(str(out_model_path))
    print("SUCESSO: df_modelagem salvo!")

    print("\nValidando leitura dos DataFrames salvos:")
    df_audit_read = spark.read.parquet(str(out_audit_path))
    df_model_read = spark.read.parquet(str(out_model_path))
    print(f"  - df_target_audit lido com sucesso: {df_audit_read.count():,} registros")
    print(f"  - df_modelagem lido com sucesso: {df_model_read.count():,} registros")

    spark.stop()
    print("SparkSession encerrada com sucesso.")

if __name__ == "__main__":
    main()
