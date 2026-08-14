import os
import sys
import shutil
import time
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = root_dir / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

UFS_NORDESTE = [21, 22, 23, 24, 25, 26, 27, 28, 29]
UF_MAP = {
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB",
    26: "PE", 27: "AL", 28: "SE", 29: "BA"
}

def main():
    print("=== INICIANDO PIPELINE NORDESTE DA ETAPA 6 EM DIANTE ===")
    spark = SparkSession.builder \
        .appName("NordestePipelineAnalysis") \
        .master("local[2]") \
        .config("spark.driver.memory", "3g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    spark.catalog.clearCache()

    monthly_dir = root_dir / "outputs" / "nordeste_monthly"
    print(f"Lendo Parquets mensais do Nordeste de: {monthly_dir}")
    
    df_ne_raw = spark.read.parquet(str(monthly_dir / "competencia=*"))
    
    # 1. VERIFICAR CONTAGEM POR ESTADO E ANO
    print("\n--- 6. CONTAGEM POR ESTADO E ANO ---")
    df_with_year = df_ne_raw.withColumn("ano", F.substring(F.col("competênciamov"), 1, 4).cast("int"))
    
    uf_counts_df = df_with_year.groupBy("uf", "ano").count().collect()
    
    uf_year_summary = {}
    for row in uf_counts_df:
        uf_code = row["uf"]
        uf_sigla = UF_MAP.get(uf_code, str(uf_code))
        ano = row["ano"]
        cnt = row["count"]
        if uf_sigla not in uf_year_summary:
            uf_year_summary[uf_sigla] = {}
        uf_year_summary[uf_sigla][ano] = cnt
        
    print("Contagem por Estado e Ano:")
    uf_totals = {}
    for sigla in sorted(UF_MAP.values()):
        c2023 = uf_year_summary.get(sigla, {}).get(2023, 0)
        c2024 = uf_year_summary.get(sigla, {}).get(2024, 0)
        c2025 = uf_year_summary.get(sigla, {}).get(2025, 0)
        tot = c2023 + c2024 + c2025
        uf_totals[sigla] = tot
        print(f"  {sigla}: 2023={c2023:,} | 2024={c2024:,} | 2025={c2025:,} | TOTAL={tot:,}")

    total_2023 = sum(uf_year_summary.get(s, {}).get(2023, 0) for s in UF_MAP.values())
    total_2024 = sum(uf_year_summary.get(s, {}).get(2024, 0) for s in UF_MAP.values())
    total_2025 = sum(uf_year_summary.get(s, {}).get(2025, 0) for s in UF_MAP.values())
    total_nordeste = total_2023 + total_2024 + total_2025

    print(f"\nTOTAL NORDESTE 2023: {total_2023:,}")
    print(f"TOTAL NORDESTE 2024: {total_2024:,}")
    print(f"TOTAL NORDESTE 2025: {total_2025:,}")
    print(f"TOTAL NORDESTE ACUMULADO: {total_nordeste:,}")
    print(f"MINIMO 500K ATENDIDO: {'SIM' if total_nordeste >= 500000 else 'NÃO'}")

    # 2. VERIFICAR COMPETÊNCIAS
    print("\n--- 7. VERIFICAR COMPETÊNCIAS ---")
    distinct_comps = sorted([r["competênciamov"] for r in df_ne_raw.select("competênciamov").distinct().collect()])
    print(f"Competências presentes ({len(distinct_comps)}): {distinct_comps}")
    
    all_possible_comps = [f"2023{m:02d}" for m in range(1, 13)] + \
                         [f"2024{m:02d}" for m in range(1, 13)] + \
                         [f"2025{m:02d}" for m in range(1, 13)]
    missing_comps = [c for c in all_possible_comps if c not in distinct_comps]
    print(f"Competências ausentes ({len(missing_comps)}): {missing_comps}")

    # 3. FEATURE PREPARATION
    print("\n--- PREPARAÇÃO DAS FEATURES DE AGRUPAMENTO ---")
    df_prep = df_ne_raw \
        .withColumn('ano', F.substring(F.col('competênciamov'), 1, 4).cast('int')) \
        .withColumn('mes', F.substring(F.col('competênciamov'), 5, 2).cast('int')) \
        .withColumn('month_seq', F.col('ano') * 12 + F.col('mes')) \
        .withColumn('FAIXA_ETARIA', 
            F.when(F.col('idade').isNull() | (F.col('idade') < 14), 'DESCONHECIDO')
             .when((F.col('idade') >= 14) & (F.col('idade') <= 17), '14-17')
             .when((F.col('idade') >= 18) & (F.col('idade') <= 24), '18-24')
             .when((F.col('idade') >= 25) & (F.col('idade') <= 34), '25-34')
             .when((F.col('idade') >= 35) & (F.col('idade') <= 44), '35-44')
             .when((F.col('idade') >= 45) & (F.col('idade') <= 54), '45-54')
             .when((F.col('idade') >= 55) & (F.col('idade') <= 64), '55-64')
             .when(F.col('idade') >= 65, '65+')
             .otherwise('DESCONHECIDO')
        ) \
        .withColumn('GRUPO_CBO', F.substring(F.col('cbo2002ocupação'), 1, 2)) \
        .withColumn('FAIXA_SALARIAL',
            F.when(F.col('salário').isNull() | (F.col('salário') <= 0), 'ATÊ_1_SM')
             .when(F.col('salário') <= 1412.0, 'ATÊ_1_SM')
             .when((F.col('salário') > 1412.0) & (F.col('salário') <= 2824.0), '1_A_2_SM')
             .when((F.col('salário') > 2824.0) & (F.col('salário') <= 4236.0), '2_A_3_SM')
             .when((F.col('salário') > 4236.0) & (F.col('salário') <= 7060.0), '3_A_5_SM')
             .when(F.col('salário') > 7060.0, 'MAIS_DE_5_SM')
             .otherwise('DESCONHECIDO')
        )

    # 4. REAVALIAR DEFINIÇÃO DE COORTE
    print("\n--- 9. REAVALIAR A DEFINIÇÃO DE COORTE NO NORDESTE ---")
    def evaluate_cohort(df, cols, name):
        grouped = df.groupBy(cols).agg(F.count('*').alias('n_total'))
        total_groups = grouped.count()
        st_mean = grouped.select(F.avg('n_total')).first()[0]
        quantiles = grouped.stat.approxQuantile('n_total', [0.25, 0.50, 0.75, 0.90], 0.01)
        c10 = grouped.filter(F.col('n_total') < 10).count()
        c20 = grouped.filter(F.col('n_total') < 20).count()
        c50 = grouped.filter(F.col('n_total') < 50).count()
        c100 = grouped.filter(F.col('n_total') < 100).count()
        
        print(f"=== {name} ===")
        print(f"  Total de Coortes: {total_groups:,}")
        print(f"  Média: {st_mean:.2f} | Mediana (P50): {quantiles[1]}")
        print(f"  P25: {quantiles[0]} | P75: {quantiles[2]} | P90: {quantiles[3]}")
        print(f"  % Coortes < 10 registros: {(c10/total_groups)*100:.2f}%")
        print(f"  % Coortes < 20 registros: {(c20/total_groups)*100:.2f}%")
        print(f"  % Coortes < 50 registros: {(c50/total_groups)*100:.2f}%")
        print(f"  % Coortes < 100 registros: {(c100/total_groups)*100:.2f}%\n")
        return {
            "name": name,
            "total_groups": total_groups,
            "mean": st_mean,
            "p25": quantiles[0],
            "p50": quantiles[1],
            "p75": quantiles[2],
            "p90": quantiles[3],
            "pct_lt_10": (c10/total_groups)*100,
            "pct_lt_20": (c20/total_groups)*100,
            "pct_lt_50": (c50/total_groups)*100,
            "pct_lt_100": (c100/total_groups)*100,
        }

    res_a = evaluate_cohort(df_prep, ['competênciamov', 'uf', 'seção', 'FAIXA_ETARIA'], 'Coorte A (Agregada)')
    res_b = evaluate_cohort(df_prep, ['competênciamov', 'uf', 'seção', 'GRUPO_CBO', 'FAIXA_ETARIA', 'graudeinstrução'], 'Coorte B (Intermediária)')
    res_c = evaluate_cohort(df_prep, ['competênciamov', 'uf', 'seção', 'GRUPO_CBO', 'FAIXA_ETARIA', 'graudeinstrução', 'sexo', 'FAIXA_SALARIAL'], 'Coorte C (Detalhada)')

    # Escolha da Coorte A
    profile_cols = ['uf', 'seção', 'FAIXA_ETARIA']
    grouped_a = df_prep.groupBy(['competênciamov'] + profile_cols).agg(F.count('*').alias('n_total'))
    total_coortes_a = grouped_a.count()
    total_recs_brutos = df_prep.count()

    print("\n--- 10. REAVALIAÇÃO DO TAMANHO MÍNIMO DE COORTE ---")
    thresh_results = {}
    for thresh in [10, 20, 50, 100]:
        filt = grouped_a.filter(F.col('n_total') >= thresh)
        rem_c = filt.count()
        pct_c_rem = ((total_coortes_a - rem_c) / total_coortes_a) * 100
        recs_rep_row = filt.select(F.sum('n_total')).first()[0]
        recs_rep = recs_rep_row if recs_rep_row is not None else 0
        pct_recs = (recs_rep / total_recs_brutos) * 100
        print(f"Limiar mínimo {thresh:3d} registros/coorte:")
        print(f"  - Coortes antes: {total_coortes_a:,} | Coortes depois: {rem_c:,} | % Removidas: {pct_c_rem:.2f}%")
        print(f"  - Registros preservados: {recs_rep:,} ({pct_recs:.2f}% do volume Nordeste)\n")
        thresh_results[thresh] = {
            "coortes_antes": total_coortes_a,
            "coortes_depois": rem_c,
            "pct_removida": pct_c_rem,
            "recs_preservados": recs_rep,
            "pct_recs_preservados": pct_recs
        }

    # Mantemos todas as coortes (limiar 1 ou integral) conforme diretrizes conservadoras, ou explicitamos escolha
    # 5. AGRUPAMENTO MENSAL t0
    print("\n--- 11. RECONSTRUIR O TARGET PARA NORDESTE ---")
    df_monthly = df_prep.groupBy(['month_seq', 'competênciamov'] + profile_cols).agg(
        F.count('*').alias('N_TOTAL_T'),
        F.sum(F.when(F.col('saldomovimentação') == 1, 1).otherwise(0)).alias('N_POSITIVOS_T'),
        F.sum(F.when(F.col('saldomovimentação') == -1, 1).otherwise(0)).alias('N_NEGATIVOS_T')
    )

    seq_map = df_monthly.select('month_seq', 'competênciamov').distinct().sort('month_seq').collect()
    seq_dict = {r['month_seq']: r['competênciamov'] for r in seq_map}

    # 12. CENSURA À DIREITA
    print("\n--- 12. CENSURA À DIREITA ---")
    max_seq = max(seq_dict.keys())
    max_comp = seq_dict[max_seq]
    eligible_max_seq = max_seq - 6
    eligible_max_comp = seq_dict[eligible_max_seq]

    print(f"Competência máxima no dataset: {max_comp} (seq: {max_seq})")
    print(f"Última competência de referência elegível (t0): {eligible_max_comp} (seq: {eligible_max_seq})")

    df_ref = df_monthly.filter(F.col('month_seq') <= eligible_max_seq)

    # 13. AUDITORIA DE LEAKAGE & RECONSTRUÇÃO DO TARGET
    df_fut = df_monthly.select(
        F.col('month_seq').alias('fut_month_seq'),
        F.col('uf').alias('fut_uf'),
        F.col('seção').alias('fut_seção'),
        F.col('FAIXA_ETARIA').alias('fut_faixa_etaria'),
        F.col('N_TOTAL_T').alias('fut_n_total'),
        F.col('N_POSITIVOS_T').alias('fut_n_positivos'),
        F.col('N_NEGATIVOS_T').alias('fut_n_negativos')
    )

    cond_join = (
        (df_ref['uf'] == df_fut['fut_uf']) &
        (df_ref['seção'] == df_fut['fut_seção']) &
        (df_ref['FAIXA_ETARIA'] == df_fut['fut_faixa_etaria']) &
        (df_fut['fut_month_seq'] > df_ref['month_seq']) &
        (df_fut['fut_month_seq'] <= df_ref['month_seq'] + 6)
    )

    joined = df_ref.join(df_fut, cond_join, 'left')

    df_target_raw = joined.groupBy(
        ['month_seq', 'competênciamov', 'N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T'] + profile_cols
    ).agg(
        F.coalesce(F.sum('fut_n_negativos'), F.lit(0)).alias('N_NEGATIVOS_6M'),
        F.coalesce(F.sum('fut_n_positivos'), F.lit(0)).alias('N_POSITIVOS_6M'),
        F.coalesce(F.sum('fut_n_total'), F.lit(0)).alias('N_TOTAL_6M')
    )

    df_target_audit = df_target_raw \
        .withColumn('PROP_NEGATIVOS_6M', 
            F.when(F.col('N_TOTAL_6M') > 0, F.col('N_NEGATIVOS_6M') / F.col('N_TOTAL_6M')).otherwise(0.0)
        )

    quantiles_ne = df_target_audit.stat.approxQuantile('PROP_NEGATIVOS_6M', [0.25, 0.50, 0.75, 0.90], 0.001)
    p50_nordeste = quantiles_ne[1]
    p75_nordeste = quantiles_ne[2]

    print(f"Quantis do Target Nordeste:")
    print(f"  P25_NORDESTE: {quantiles_ne[0]:.6f}")
    print(f"  P50_NORDESTE (Mediana): {p50_nordeste:.6f}")
    print(f"  P75_NORDESTE: {p75_nordeste:.6f}")

    df_target_audit = df_target_audit \
        .withColumn('ALTA_ROTATIVIDADE_6M', F.when(F.col('PROP_NEGATIVOS_6M') > p50_nordeste, 1).otherwise(0))

    coortes_elegiveis_total = df_target_audit.count()
    class_dist = df_target_audit.groupBy('ALTA_ROTATIVIDADE_6M').agg(
        F.count('*').alias('cnt')
    ).collect()

    class_counts = {r['ALTA_ROTATIVIDADE_6M']: r['cnt'] for r in class_dist}
    pct_c0 = (class_counts.get(0, 0) / coortes_elegiveis_total) * 100
    pct_c1 = (class_counts.get(1, 0) / coortes_elegiveis_total) * 100

    print(f"Distribuição de Classes:")
    print(f"  Classe 0: {class_counts.get(0, 0):,} ({pct_c0:.2f}%)")
    print(f"  Classe 1: {class_counts.get(1, 0):,} ({pct_c1:.2f}%)")

    # Separar df_modelagem (Sem leakage)
    df_modelagem = df_target_audit.select(
        'competênciamov', 'uf', 'seção', 'FAIXA_ETARIA',
        'N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T',
        'ALTA_ROTATIVIDADE_6M'
    )

    print("\n--- 14. SALVAR OS INTERMEDIÁRIOS DO NORDESTE ---")
    out_audit_dir = root_dir / "outputs" / "target_audit_nordeste"
    out_audit_dir.mkdir(parents=True, exist_ok=True)

    path_audit_parquet = out_audit_dir / "df_target_audit.parquet"
    path_model_parquet = out_audit_dir / "df_modelagem.parquet"

    if path_audit_parquet.exists(): shutil.rmtree(path_audit_parquet)
    if path_model_parquet.exists(): shutil.rmtree(path_model_parquet)

    print(f"Gravando df_target_audit.parquet em: {path_audit_parquet}")
    df_target_audit.write.mode("overwrite").parquet(str(path_audit_parquet))

    print(f"Gravando df_modelagem.parquet em: {path_model_parquet}")
    df_modelagem.write.mode("overwrite").parquet(str(path_model_parquet))

    # Validação da gravação
    df_audit_check = spark.read.parquet(str(path_audit_parquet))
    df_model_check = spark.read.parquet(str(path_model_parquet))

    count_audit = df_audit_check.count()
    count_model = df_model_check.count()

    has_audit_success = (path_audit_parquet / "_SUCCESS").exists()
    has_model_success = (path_model_parquet / "_SUCCESS").exists()

    print("\nValidação de Gravação:")
    print(f"  df_target_audit: count={count_audit:,} | _SUCCESS={has_audit_success}")
    print(f"  df_modelagem: count={count_model:,} | _SUCCESS={has_model_success}")

    results_summary = {
        "spark_master": "local[2]",
        "driver_memory": "3g",
        "shuffle_partitions": "32",
        "ufs_nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
        "ufs_codigos": [27, 29, 23, 21, 25, 26, 22, 24, 28],
        "competencias_processadas": distinct_comps,
        "competencias_ausentes": missing_comps,
        "uf_totals": uf_totals,
        "total_2023": total_2023,
        "total_2024": total_2024,
        "total_2025": total_2025,
        "total_nordeste": total_nordeste,
        "minimo_500k_atendido": total_nordeste >= 500000,
        "coorte_escolhida": "Coorte A (competênciamov + uf + seção + FAIXA_ETARIA)",
        "numero_coortes": total_coortes_a,
        "tamanho_minimo_coorte": 1, # integral preservada sem descarte prematuro
        "ultima_competencia_elegivel": eligible_max_comp,
        "p50_nordeste": p50_nordeste,
        "p75_nordeste": p75_nordeste,
        "class_0_pct": pct_c0,
        "class_1_pct": pct_c1,
        "df_target_audit_salvo": has_audit_success and count_audit == coortes_elegiveis_total,
        "df_modelagem_salvo": has_model_success and count_model == coortes_elegiveis_total,
        "caminho_parquets": str(out_audit_dir),
        "coorte_eval_a": res_a,
        "coorte_eval_b": res_b,
        "coorte_eval_c": res_c,
        "thresh_results": thresh_results
    }

    with open(root_dir / "outputs" / "nordeste_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\nSUCESSO: Pipeline Nordeste executado e resumido em outputs/nordeste_summary.json!")
    spark.stop()

if __name__ == "__main__":
    main()
