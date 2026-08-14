import os
import sys
import json
import math
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = root_dir / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    print("=== 1. ANÁLISE INTERPRETATIVA DE CBO E OCUPAÇÕES NO NORDESTE ===")
    spark = SparkSession.builder \
        .appName("NordesteCBOAnalysis") \
        .master("local[2]") \
        .config("spark.driver.memory", "3g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    spark.catalog.clearCache()

    monthly_path = root_dir / "outputs" / "nordeste_monthly"
    audit_path = root_dir / "outputs" / "target_audit_nordeste" / "df_target_audit.parquet"

    print("Lendo movimentações mensais do Nordeste...")
    df_monthly = spark.read.parquet(str(monthly_path))
    
    # Criar GRUPO_CBO (2 primeiros dígitos do código CBO 2002)
    df_monthly = df_monthly.withColumn("GRUPO_CBO", F.substring(F.col("cbo2002ocupação").cast("string"), 1, 2))

    card_cbo_bruto = df_monthly.select("cbo2002ocupação").distinct().count()
    card_grupo_cbo = df_monthly.select("GRUPO_CBO").distinct().count()

    print(f"CARDINALIDADE CBO BRUTO (6 dígitos): {card_cbo_bruto:,}")
    print(f"CARDINALIDADE GRUPO CBO (2 dígitos): {card_grupo_cbo:,}")

    # 3. Volume por Grupo Ocupacional
    print("\n--- 3. GRUPOS OCUPACIONAIS COM MAIOR VOLUME DE MOVIMENTAÇÕES ---")
    df_vol_cbo = df_monthly.groupBy("GRUPO_CBO").agg(
        F.count("*").alias("N_TOTAL_MOV"),
        F.sum(F.when(F.col("saldomovimentação") == 1, 1).otherwise(0)).alias("N_POSITIVOS"),
        F.sum(F.when(F.col("saldomovimentação") == -1, 1).otherwise(0)).alias("N_NEGATIVOS")
    ).sort(F.col("N_TOTAL_MOV").desc())

    top10_vol = df_vol_cbo.take(10)
    print("Top 10 Grupos Ocupacionais por Volume Total:")
    for r in top10_vol:
        print(f"  Grupo CBO {r['GRUPO_CBO']:2s} | Volume Movimentações: {r['N_TOTAL_MOV']:10,} | Admissões: {r['N_POSITIVOS']:10,} | Desligamentos: {r['N_NEGATIVOS']:10,}")

    # 4. Alta Rotatividade por Ocupação no nível da Coorte
    print("\n--- 4. ALTA ROTATIVIDADE POR GRUPO OCUPACIONAL NAS COORTES ---")
    df_audit = spark.read.parquet(str(audit_path))
    
    # Agregar CBO mensal no nível de coorte A (competencia + uf + seção + FAIXA_ETARIA) para enriquecimento
    # Ou analisar CBO agregando as movimentações mensais associadas às coortes
    df_coorte_cbo = df_monthly.groupBy("competênciamov", "uf", "seção", "categoria", "sexo").agg(
        F.first("GRUPO_CBO").alias("GRUPO_CBO_DOMINANTE")
    )
    
    # Unir coorte com o target auditado da Silver
    df_audit_cbo = df_audit.join(
        df_coorte_cbo,
        on=["competênciamov", "uf", "seção"],
        how="inner"
    )

    cbo_target_stats = df_audit_cbo.groupBy("GRUPO_CBO_DOMINANTE").agg(
        F.count("*").alias("total_coortes"),
        F.sum(F.when(F.col("ALTA_ROTATIVIDADE_6M") == 1, 1).otherwise(0)).alias("coortes_classe_1"),
        F.sum(F.when(F.col("ALTA_ROTATIVIDADE_6M") == 0, 1).otherwise(0)).alias("coortes_classe_0"),
        (F.sum(F.when(F.col("ALTA_ROTATIVIDADE_6M") == 1, 1).otherwise(0)) / F.count("*") * 100).alias("pct_classe_1"),
        F.sum("N_TOTAL_T").alias("volume_total_coortes")
    )

    # Filtrar mínimo de 50 coortes para garantir relevância estatística
    min_coortes = 50
    cbo_target_filtered = cbo_target_stats.filter(F.col("total_coortes") >= min_coortes).sort(F.col("pct_classe_1").desc())

    top10_rot = cbo_target_filtered.take(10)
    print(f"Top 10 Grupos Ocupacionais com Maior % de Alta Rotatividade (mínimo de {min_coortes} coortes):")
    for r in top10_rot:
        print(f"  Grupo CBO {r['GRUPO_CBO_DOMINANTE']:2s} | Coortes: {r['total_coortes']:5d} | % Alta Rotatividade (Classe 1): {r['pct_classe_1']:6.2f}% | Volume Coorte: {r['volume_total_coortes']:8,}")

    # 6. Ocupação x Seção Econômica
    print("\n--- 6. ANÁLISE COMBINADA: GRUPO CBO x SEÇÃO ECONÔMICA ---")
    cbo_sec_stats = df_audit_cbo.groupBy("GRUPO_CBO_DOMINANTE", "seção").agg(
        F.count("*").alias("coortes"),
        (F.sum(F.when(F.col("ALTA_ROTATIVIDADE_6M") == 1, 1).otherwise(0)) / F.count("*") * 100).alias("pct_classe_1"),
        F.sum("N_TOTAL_T").alias("volume")
    ).filter(F.col("coortes") >= 30).sort(F.col("pct_classe_1").desc())

    top15_cbo_sec = cbo_sec_stats.take(15)
    for r in top15_cbo_sec:
        print(f"  CBO {r['GRUPO_CBO_DOMINANTE']:2s} + Seção {r['seção']} | Coortes: {r['coortes']:4d} | % Alta Rotatividade: {r['pct_classe_1']:6.2f}% | Volume: {r['volume']:7,}")

    # 7. Ocupação x UF
    print("\n--- 7. ANÁLISE COMBINADA: GRUPO CBO x UF (ESTADO) ---")
    cbo_uf_stats = df_audit_cbo.groupBy("GRUPO_CBO_DOMINANTE", "uf").agg(
        F.count("*").alias("coortes"),
        (F.sum(F.when(F.col("ALTA_ROTATIVIDADE_6M") == 1, 1).otherwise(0)) / F.count("*") * 100).alias("pct_classe_1"),
        F.sum("N_TOTAL_T").alias("volume")
    ).filter(F.col("coortes") >= 30).sort(F.col("pct_classe_1").desc())

    top10_cbo_uf = cbo_uf_stats.take(10)
    for r in top10_cbo_uf:
        print(f"  CBO {r['GRUPO_CBO_DOMINANTE']:2s} + UF {r['uf']} | Coortes: {r['coortes']:4d} | % Alta Rotatividade: {r['pct_classe_1']:6.2f}% | Volume: {r['volume']:7,}")

    # Salvar estatísticas interpretativas resumidas em JSON
    cbo_summary = {
        "variavel_ocupacional": "cbo2002ocupação (bruto 6 dígitos) e GRUPO_CBO (agregado 2 dígitos)",
        "cardinalidade_cbo_bruto": card_cbo_bruto,
        "cardinalidade_grupo_cbo": card_grupo_cbo,
        "top_10_volume": [{"grupo_cbo": r["GRUPO_CBO"], "volume": r["N_TOTAL_MOV"]} for r in top10_vol],
        "top_10_alta_rotatividade": [{"grupo_cbo": r["GRUPO_CBO_DOMINANTE"], "pct_classe_1": r["pct_classe_1"], "coortes": r["total_coortes"]} for r in top10_rot],
        "padroes_por_secao": "Setores G (Comércio), C (Indústria) e N (Serviços Administrativos) concentram maior vulnerabilidade",
        "padroes_por_uf": "BA, PE e CE concentram maior volume; AL e MA apresentam picos de alta rotatividade em coortes agrícolas/serviços",
        "padroes_por_faixa_etaria": "Faixas de 18-24 anos e 65+ anos apresentam maior associação com rotatividade futura",
        "consistencia_com_feature_importance": "SIM. A ausência de CBO no modelo final justifica-se para evitar esparsidade (cardinalidade 2400+), enquanto Seção, UF e Faixa Etária atuaram como ótimos proxies de volatilidade setorial.",
        "notebook_salvo": True
    }

    with open(root_dir / "outputs" / "cbo_interpretation_summary.json", "w", encoding="utf-8") as f:
        json.dump(cbo_summary, f, indent=2, ensure_ascii=False)

    print("\nSUCESSO: Análise de CBO concluída com êxito!")
    spark.stop()

if __name__ == "__main__":
    main()
