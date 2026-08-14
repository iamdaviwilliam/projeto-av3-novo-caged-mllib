import os
import sys
import shutil
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = root_dir / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

UFS_NORDESTE = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# MA:21, PI:22, CE:23, RN:24, PB:25, PE:26, AL:27, SE:28, BA:29

def main():
    start_time = time.time()
    spark = SparkSession.builder \
        .appName("NordesteMonthlyIngestion") \
        .master("local[2]") \
        .config("spark.driver.memory", "3g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    spark.catalog.clearCache()

    raw_dir = root_dir / "data" / "raw" / "extracted"
    out_dir = root_dir / "outputs" / "nordeste_monthly"

    files = sorted(list(raw_dir.rglob("CAGEDMOV*.txt")))
    print(f"Encontrados {len(files)} arquivos mensais CAGEDMOV.")

    total_records_processed = 0

    for idx, fpath in enumerate(files):
        comp = fpath.stem.replace("CAGEDMOV", "")
        month_out_dir = out_dir / f"competencia={comp}"
        
        # Leitura mensal com filtro IMEDIATO no Nordeste
        df_month_raw = spark.read \
            .option("header", "true") \
            .option("delimiter", ";") \
            .option("encoding", "utf-8") \
            .csv(str(fpath))

        # Normalizar nomes de colunas
        col_renames = {c: c.strip().lower() for c in df_month_raw.columns}
        df_month = df_month_raw
        for old_c, new_c in col_renames.items():
            df_month = df_month.withColumnRenamed(old_c, new_c)

        # Filtro Nordeste o mais cedo possível
        df_ne = df_month.filter(F.col("uf").cast("int").isin(UFS_NORDESTE))

        # Seleção de colunas necessárias + transformações mínimas
        salario_col = "salário" if "salário" in df_ne.columns else "valorsaláriofixo"
        
        df_reduced = df_ne.select(
            F.lit(comp).alias("competênciamov"),
            F.col("uf").cast("int").alias("uf"),
            F.col("seção").alias("seção"),
            F.col("idade").cast("int").alias("idade"),
            F.col("graudeinstrução").cast("int").alias("graudeinstrução"),
            F.col("cbo2002ocupação").alias("cbo2002ocupação"),
            F.col("sexo").cast("int").alias("sexo"),
            F.regexp_replace(F.col(salario_col), ",", ".").cast("double").alias("salário"),
            F.col("saldomovimentação").cast("int").alias("saldomovimentação"),
            F.col("categoria").cast("int").alias("categoria")
        ).filter(
            F.col("uf").isNotNull() & 
            F.col("seção").isNotNull() & 
            F.col("saldomovimentação").isin(1, -1)
        )

        if month_out_dir.exists():
            shutil.rmtree(month_out_dir)

        df_reduced.write.mode("overwrite").parquet(str(month_out_dir))
        
        # Contagem para verificação
        rec_count = spark.read.parquet(str(month_out_dir)).count()
        total_records_processed += rec_count

        print(f"[{idx+1:02d}/{len(files)}] Competência {comp}: {rec_count:,} registros salvos em {month_out_dir.name}")

        # Liberar referências
        df_month_raw.unpersist()
        df_month.unpersist()
        df_ne.unpersist()
        df_reduced.unpersist()

    print(f"\nProcessamento concluído em {time.time() - start_time:.2f}s.")
    print(f"Total de registros do Nordeste salvos (2023-2025): {total_records_processed:,}")

    spark.stop()

if __name__ == "__main__":
    main()
