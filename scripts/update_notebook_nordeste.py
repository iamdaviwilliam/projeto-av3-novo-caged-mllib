import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
nb_path = root_dir / "notebooks" / "pipeline_mllib.ipynb"
summary_path = root_dir / "outputs" / "nordeste_summary.json"

def main():
    print(f"Lendo resumo dos resultados do Nordeste de: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print(f"Lendo notebook original de: {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Construção das novas células para a Seção de Adequação do Escopo (Nordeste)
    nordeste_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Adequação do Escopo — Região Nordeste\n",
                "\n",
                "### 8.1 Motivação e Justificativa de Engenharia de Dados\n",
                "- **Diagnóstico de Estabilidade Local (8 GB de RAM):** O processamento da base bruta nacional do Novo CAGED para o triênio 2023–2025 engloba mais de 118 milhões de registros de movimentação. Durante execuções locais em máquina com 8 GB de memoria RAM, o volume nacional gerou instabilidade operacional e estouro de memória no driver/executor do PySpark.\n",
                "- **Adequação ao Roteiro Metodológico:** Conforme previsto nas orientações da disciplina e roteiro do projeto, é permitida e recomendada a limitação do escopo geográfico a uma região do país para viabilidade computacional local.\n",
                "- **Escopo Selecionado:** Restrição dos dados à **Região Nordeste** do Brasil, composta por 9 Unidades da Federação (AL, BA, CE, MA, PB, PE, PI, RN, SE), mantendo integralmente a janela temporal histórica de **2023 a 2025**.\n",
                "- **Estratégia de Ingestão Eficiente (Filtro Precoce):** Para garantir o consumo conservador de memória, a filtragem geográfica dos estados nordestinos passa a ocorrer **imediatamente após a leitura individual de cada arquivo mensal**, seguida da seleção estrita de colunas e escrita intermediária em disco (`outputs/nordeste_monthly/`).\n",
                "- **Preservação do Histórico Exploratório:** As seções nacionais anteriores são mantidas no notebook como análise exploratória inicial contextual, enquanto toda a etapa de construção de coortes, target e modelagem final passa a ser fundamentada exclusivamente no universo da Região Nordeste.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8.2 Configuração Conservadora do PySpark\n",
                "\n",
                "Utilização de parâmetros conservadores de execução Spark (`local[2]`, `3g` driver memory, 32 shuffle partitions e AQE habilitado) sem a utilização de `local[*]`, limpando caches anteriores via `spark.catalog.clearCache()`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pyspark.sql import SparkSession\n",
                "import pyspark.sql.functions as F\n",
                "\n",
                "spark = SparkSession.builder \\\n",
                "    .appName('CagedNordestePipeline') \\\n",
                "    .master('local[2]') \\\n",
                "    .config('spark.driver.memory', '3g') \\\n",
                "    .config('spark.sql.shuffle.partitions', '32') \\\n",
                "    .config('spark.sql.adaptive.enabled', 'true') \\\n",
                "    .config('spark.sql.execution.arrow.pyspark.enabled', 'true') \\\n",
                "    .getOrCreate()\n",
                "\n",
                "spark.catalog.clearCache()\n",
                "print(\"SparkSession reconfigurada conservadoramente com sucesso:\")\n",
                "print(f\"  - Master: local[2]\")\n",
                "print(f\"  - Driver Memory: 3g\")\n",
                "print(f\"  - Shuffle Partitions: 32\")\n",
                "print(f\"  - Adaptive Query Execution (AQE): true\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8.3 Mapeamento Explícito das UFs da Região Nordeste\n",
                "\n",
                "Inspeção prévia dos microdados confirmou que a coluna `uf` utiliza **códigos numéricos do IBGE** (e não siglas de duas letras). A Região Nordeste foi mapeada explicitamente da seguinte forma:\n",
                "\n",
                "| Sigla UF | Código IBGE | Estado |\n",
                "|---|---|---|\n",
                "| MA | 21 | Maranhão |\n",
                "| PI | 22 | Piauí |\n",
                "| CE | 23 | Ceará |\n",
                "| RN | 24 | Rio Grande do Norte |\n",
                "| PB | 25 | Paraíba |\n",
                "| PE | 26 | Pernambuco |\n",
                "| AL | 27 | Alagoas |\n",
                "| SE | 28 | Sergipe |\n",
                "| BA | 29 | Bahia |\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Definição explícita dos códigos IBGE dos 9 estados do Nordeste\n",
                "UFS_NORDESTE = [21, 22, 23, 24, 25, 26, 27, 28, 29]\n",
                "UF_MAP = {\n",
                "    21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB',\n",
                "    26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA'\n",
                "}\n",
                "print(f\"UFS_NORDESTE (Códigos IBGE): {UFS_NORDESTE}\")\n",
                "print(f\"Mapeamento de Siglas: {UF_MAP}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8.4 Ingestão Mensal com Filtragem Precoce e Persistência Intermediária Parquet\n",
                "\n",
                "Processamento mês a mês para reduzir o uso de memória RAM. Cada arquivo mensal é lido, filtrado no Nordeste imediatamente, limpo nas colunas estritamente necessárias e gravado em `outputs/nordeste_monthly/competencia=YYYYMM/`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Leitura dos Parquets reduzidos da camada intermediária do Nordeste\n",
                "df_ne_raw = spark.read.parquet('../outputs/nordeste_monthly/competencia=*')\n",
                "total_recs = df_ne_raw.count()\n",
                "print(f\"Total de movimentações carregadas do Nordeste (2023-2025): {total_recs:,}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8.5 Auditoria de Movimentações por Estado, Ano e Verificação de Competências\n",
                "\n",
                "Verificação da cobertura temporal (2023–2025) e contagens por Estado e Ano."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Contagem por UF e Ano\n",
                "df_with_year = df_ne_raw.withColumn('ano', F.substring(F.col('competênciamov'), 1, 4).cast('int'))\n",
                "uf_counts = df_with_year.groupBy('uf', 'ano').count().collect()\n",
                "\n",
                "uf_summary = {}\n",
                "for r in uf_counts:\n",
                "    sigla = UF_MAP.get(r['uf'], str(r['uf']))\n",
                "    ano = r['ano']\n",
                "    uf_summary.setdefault(sigla, {})[ano] = r['count']\n",
                "\n",
                "print(\"=== MOVIMENTAÇÕES POR ESTADO E ANO (NORDESTE) ===\")\n",
                "for sigla in sorted(UF_MAP.values()):\n",
                "    c23 = uf_summary.get(sigla, {}).get(2023, 0)\n",
                "    c24 = uf_summary.get(sigla, {}).get(2024, 0)\n",
                "    c25 = uf_summary.get(sigla, {}).get(2025, 0)\n",
                "    tot = c23 + c24 + c25\n",
                "    print(f\"  {sigla}: 2023 = {c23:9,} | 2024 = {c24:9,} | 2025 = {c25:9,} | TOTAL = {tot:10,}\")\n",
                "\n",
                "tot_23 = sum(uf_summary.get(s, {}).get(2023, 0) for s in UF_MAP.values())\n",
                "tot_24 = sum(uf_summary.get(s, {}).get(2024, 0) for s in UF_MAP.values())\n",
                "tot_25 = sum(uf_summary.get(s, {}).get(2025, 0) for s in UF_MAP.values())\n",
                "tot_ne = tot_23 + tot_24 + tot_25\n",
                "\n",
                "print(f\"\\nTOTAL NORDESTE 2023: {tot_23:,}\")\n",
                "print(f\"TOTAL NORDESTE 2024: {tot_24:,}\")\n",
                "print(f\"TOTAL NORDESTE 2025: {tot_25:,}\")\n",
                "print(f\"TOTAL NORDESTE ACUMULADO: {tot_ne:,}\")\n",
                "print(f\"CRITÉRIO MÍNIMO DE 500 MIL REGISTROS ATENDIDO? {'SIM' if tot_ne >= 500000 else 'NÃO'}\")\n",
                "\n",
                "# Auditoria de Competências\n",
                "comps_presentes = sorted([r['competênciamov'] for r in df_ne_raw.select('competênciamov').distinct().collect()])\n",
                "comps_esperadas = [f'2023{m:02d}' for m in range(1, 13)] + [f'2024{m:02d}' for m in range(1, 13)] + [f'2025{m:02d}' for m in range(1, 13)]\n",
                "comps_ausentes = [c for c in comps_esperadas if c not in comps_presentes]\n",
                "\n",
                "print(f\"\\nCompetências Processadas ({len(comps_presentes)}): {comps_presentes}\")\n",
                "print(f\"Competências Ausentes ({len(comps_ausentes)}): {comps_ausentes}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8.6 Reconstrução e Reavaliação das Coortes no Universo Nordeste\n",
                "\n",
                "Comparação empírica das granularidades A, B e C utilizando dados exclusivos da Região Nordeste."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Preparação das features de agrupamento para a análise de coortes\n",
                "df_prep = df_ne_raw \\\n",
                "    .withColumn('ano', F.substring(F.col('competênciamov'), 1, 4).cast('int')) \\\n",
                "    .withColumn('mes', F.substring(F.col('competênciamov'), 5, 2).cast('int')) \\\n",
                "    .withColumn('month_seq', F.col('ano') * 12 + F.col('mes')) \\\n",
                "    .withColumn('FAIXA_ETARIA', \n",
                "        F.when(F.col('idade').isNull() | (F.col('idade') < 14), 'DESCONHECIDO')\n",
                "         .when((F.col('idade') >= 14) & (F.col('idade') <= 17), '14-17')\n",
                "         .when((F.col('idade') >= 18) & (F.col('idade') <= 24), '18-24')\n",
                "         .when((F.col('idade') >= 25) & (F.col('idade') <= 34), '25-34')\n",
                "         .when((F.col('idade') >= 35) & (F.col('idade') <= 44), '35-44')\n",
                "         .when((F.col('idade') >= 45) & (F.col('idade') <= 54), '45-54')\n",
                "         .when((F.col('idade') >= 55) & (F.col('idade') <= 64), '55-64')\n",
                "         .when(F.col('idade') >= 65, '65+')\n",
                "         .otherwise('DESCONHECIDO')\n",
                "    ) \\\n",
                "    .withColumn('GRUPO_CBO', F.substring(F.col('cbo2002ocupação'), 1, 2)) \\\n",
                "    .withColumn('FAIXA_SALARIAL',\n",
                "        F.when(F.col('salário').isNull() | (F.col('salário') <= 0), 'ATÊ_1_SM')\n",
                "         .when(F.col('salário') <= 1412.0, 'ATÊ_1_SM')\n",
                "         .when((F.col('salário') > 1412.0) & (F.col('salário') <= 2824.0), '1_A_2_SM')\n",
                "         .when((F.col('salário') > 2824.0) & (F.col('salário') <= 4236.0), '2_A_3_SM')\n",
                "         .when((F.col('salário') > 4236.0) & (F.col('salário') <= 7060.0), '3_A_5_SM')\n",
                "         .when(F.col('salário') > 7060.0, 'MAIS_DE_5_SM')\n",
                "         .otherwise('DESCONHECIDO')\n",
                "    )\n",
                "\n",
                "def evaluate_cohort_ne(df, cols, name):\n",
                "    grouped = df.groupBy(cols).agg(F.count('*').alias('n_total'))\n",
                "    total_groups = grouped.count()\n",
                "    st_mean = grouped.select(F.avg('n_total')).first()[0]\n",
                "    quantiles = grouped.stat.approxQuantile('n_total', [0.25, 0.50, 0.75, 0.90], 0.01)\n",
                "    c10 = grouped.filter(F.col('n_total') < 10).count()\n",
                "    c20 = grouped.filter(F.col('n_total') < 20).count()\n",
                "    c50 = grouped.filter(F.col('n_total') < 50).count()\n",
                "    c100 = grouped.filter(F.col('n_total') < 100).count()\n",
                "    \n",
                "    print(f\"=== {name} ===\")\n",
                "    print(f\"  Total de Coortes: {total_groups:,}\")\n",
                "    print(f\"  Média: {st_mean:.2f} | Mediana (P50): {quantiles[1]}\")\n",
                "    print(f\"  P25: {quantiles[0]} | P75: {quantiles[2]} | P90: {quantiles[3]}\")\n",
                "    print(f\"  % Coortes < 10 registros: {(c10/total_groups)*100:.2f}%\")\n",
                "    print(f\"  % Coortes < 20 registros: {(c20/total_groups)*100:.2f}%\")\n",
                "    print(f\"  % Coortes < 50 registros: {(c50/total_groups)*100:.2f}%\")\n",
                "    print(f\"  % Coortes < 100 registros: {(c100/total_groups)*100:.2f}%\\n\")\n",
                "\n",
                "evaluate_cohort_ne(df_prep, ['competênciamov', 'uf', 'seção', 'FAIXA_ETARIA'], 'Coorte A (Agregada)')\n",
                "evaluate_cohort_ne(df_prep, ['competênciamov', 'uf', 'seção', 'GRUPO_CBO', 'FAIXA_ETARIA', 'graudeinstrução'], 'Coorte B (Intermediária)')\n",
                "evaluate_cohort_ne(df_prep, ['competênciamov', 'uf', 'seção', 'GRUPO_CBO', 'FAIXA_ETARIA', 'graudeinstrução', 'sexo', 'FAIXA_SALARIAL'], 'Coorte C (Detalhada)')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8.7 Reavaliação de Limiares Mínimos de Tamanho de Coorte\n",
                "\n",
                "Análise comparativa de limiares mínimos (10, 20, 50, 100) para a Coorte A no Nordeste."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "profile_cols = ['uf', 'seção', 'FAIXA_ETARIA']\n",
                "grouped_a = df_prep.groupBy(['competênciamov'] + profile_cols).agg(F.count('*').alias('n_total'))\n",
                "total_coortes_a = grouped_a.count()\n",
                "total_recs_brutos = df_prep.count()\n",
                "\n",
                "print(f\"Total de coortes brutas na Coorte A (Nordeste): {total_coortes_a:,}\")\n",
                "print(f\"Total de registros de movimentação: {total_recs_brutos:,}\\n\")\n",
                "\n",
                "print(\"=== SIMULAÇÃO DE LIMIARES MÍNIMOS DE TAMANHO DE COORTE (NORDESTE) ===\")\n",
                "for thresh in [10, 20, 50, 100]:\n",
                "    filt = grouped_a.filter(F.col('n_total') >= thresh)\n",
                "    rem_c = filt.count()\n",
                "    pct_c_rem = ((total_coortes_a - rem_c) / total_coortes_a) * 100\n",
                "    recs_rep = filt.select(F.sum('n_total')).first()[0]\n",
                "    recs_rep = recs_rep if recs_rep is not None else 0\n",
                "    pct_recs = (recs_rep / total_recs_brutos) * 100\n",
                "    print(f\"Limiar mínimo {thresh:3d} registros/coorte:\")\n",
                "    print(f\"  - Coortes mantidas: {rem_c:7,} | Removidas: {pct_c_rem:6.2f}%\")\n",
                "    print(f\"  - Registros representados: {recs_rep:11,} ({pct_recs:5.2f}% do volume Nordeste)\\n\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8.8 Reconstrução do Target Agregado no Universo Nordeste\n",
                "\n",
                "Cálculo das movimentações futuras ($t+1 \\dots t+6$), determinação da censura à direita ($t_0 \\le 202506$) e cálculo do ponto de corte mediano ($P_{50}$) exclusivo do Nordeste."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Agrupamento mensal t0\n",
                "df_monthly = df_prep.groupBy(['month_seq', 'competênciamov'] + profile_cols).agg(\n",
                "    F.count('*').alias('N_TOTAL_T'),\n",
                "    F.sum(F.when(F.col('saldomovimentação') == 1, 1).otherwise(0)).alias('N_POSITIVOS_T'),\n",
                "    F.sum(F.when(F.col('saldomovimentação') == -1, 1).otherwise(0)).alias('N_NEGATIVOS_T')\n",
                ")\n",
                "\n",
                "seq_map = df_monthly.select('month_seq', 'competênciamov').distinct().sort('month_seq').collect()\n",
                "seq_dict = {r['month_seq']: r['competênciamov'] for r in seq_map}\n",
                "\n",
                "max_seq = max(seq_dict.keys())\n",
                "max_comp = seq_dict[max_seq]\n",
                "eligible_max_seq = max_seq - 6\n",
                "eligible_max_comp = seq_dict[eligible_max_seq]\n",
                "\n",
                "print(f\"Competência máxima no dataset Nordeste: {max_comp} (seq: {max_seq})\")\n",
                "print(f\"Última competência de referência elegível (t0): {eligible_max_comp} (seq: {eligible_max_seq})\")\n",
                "\n",
                "df_ref = df_monthly.filter(F.col('month_seq') <= eligible_max_seq)\n",
                "\n",
                "# Associações de 6 meses futuros (Join de perfis socioeconômicos)\n",
                "df_fut = df_monthly.select(\n",
                "    F.col('month_seq').alias('fut_month_seq'),\n",
                "    F.col('uf').alias('fut_uf'),\n",
                "    F.col('seção').alias('fut_seção'),\n",
                "    F.col('FAIXA_ETARIA').alias('fut_faixa_etaria'),\n",
                "    F.col('N_TOTAL_T').alias('fut_n_total'),\n",
                "    F.col('N_POSITIVOS_T').alias('fut_n_positivos'),\n",
                "    F.col('N_NEGATIVOS_T').alias('fut_n_negativos')\n",
                ")\n",
                "\n",
                "cond_join = (\n",
                "    (df_ref['uf'] == df_fut['fut_uf']) &\n",
                "    (df_ref['seção'] == df_fut['fut_seção']) &\n",
                "    (df_ref['FAIXA_ETARIA'] == df_fut['fut_faixa_etaria']) &\n",
                "    (df_fut['fut_month_seq'] > df_ref['month_seq']) &\n",
                "    (df_fut['fut_month_seq'] <= df_ref['month_seq'] + 6)\n",
                ")\n",
                "\n",
                "joined = df_ref.join(df_fut, cond_join, 'left')\n",
                "\n",
                "df_target_raw = joined.groupBy(\n",
                "    ['month_seq', 'competênciamov', 'N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T'] + profile_cols\n",
                ").agg(\n",
                "    F.coalesce(F.sum('fut_n_negativos'), F.lit(0)).alias('N_NEGATIVOS_6M'),\n",
                "    F.coalesce(F.sum('fut_n_positivos'), F.lit(0)).alias('N_POSITIVOS_6M'),\n",
                "    F.coalesce(F.sum('fut_n_total'), F.lit(0)).alias('N_TOTAL_6M')\n",
                ")\n",
                "\n",
                "df_target_audit = df_target_raw \\\n",
                "    .withColumn('PROP_NEGATIVOS_6M', \n",
                "        F.when(F.col('N_TOTAL_6M') > 0, F.col('N_NEGATIVOS_6M') / F.col('N_TOTAL_6M')).otherwise(0.0)\n",
                "    )\n",
                "\n",
                "quantiles_ne = df_target_audit.stat.approxQuantile('PROP_NEGATIVOS_6M', [0.25, 0.50, 0.75, 0.90], 0.001)\n",
                "p50_nordeste = quantiles_ne[1]\n",
                "p75_nordeste = quantiles_ne[2]\n",
                "\n",
                "print(f\"\\nQuantis do Target Nordeste (PROP_NEGATIVOS_6M):\")\n",
                "print(f\"  P25_NORDESTE: {quantiles_ne[0]:.6f}\")\n",
                "print(f\"  P50_NORDESTE (Mediana): {p50_nordeste:.6f}\")\n",
                "print(f\"  P75_NORDESTE: {p75_nordeste:.6f}\")\n",
                "\n",
                "# Criação da coluna binária do target com base no P50_NORDESTE\n",
                "df_target_audit = df_target_audit \\\n",
                "    .withColumn('ALTA_ROTATIVIDADE_6M', F.when(F.col('PROP_NEGATIVOS_6M') > p50_nordeste, 1).otherwise(0))\n",
                "\n",
                "print(\"\\n=== DISTRIBUIÇÃO FINAL DAS CLASSES NO NORDESTE ===\")\n",
                "tot_coortes = df_target_audit.count()\n",
                "df_target_audit.groupBy('ALTA_ROTATIVIDADE_6M').agg(\n",
                "    F.count('*').alias('quantidade'),\n",
                "    (F.count('*') / tot_coortes * 100).alias('percentual')\n",
                ").sort('ALTA_ROTATIVIDADE_6M').show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8.9 Auditoria de Leakage e Persistência Intermediária (`df_target_audit` e `df_modelagem`)\n",
                "\n",
                "Separação formal entre o DataFrame de auditoria e o DataFrame de modelagem (isento de variáveis futuras) com persistência em Parquet."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Separação do DataFrame Lógico de Modelagem (Sem variáveis futuras)\n",
                "df_modelagem = df_target_audit.select(\n",
                "    'competênciamov', 'uf', 'seção', 'FAIXA_ETARIA',\n",
                "    'N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T',\n",
                "    'ALTA_ROTATIVIDADE_6M'\n",
                ")\n",
                "\n",
                "forbidden_cols = ['N_NEGATIVOS_6M', 'N_POSITIVOS_6M', 'N_TOTAL_6M', 'PROP_NEGATIVOS_6M']\n",
                "leakage_found = [c for c in forbidden_cols if c in df_modelagem.columns]\n",
                "print(f\"Variáveis futuras encontradas em df_modelagem: {leakage_found} (ESPERADO: [])\")\n",
                "\n",
                "# Persistência dos DataFrames intermediários do Nordeste em Parquet\n",
                "import shutil\n",
                "out_audit_dir = root_dir / 'outputs' / 'target_audit_nordeste'\n",
                "out_audit_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "p_audit = out_audit_dir / 'df_target_audit.parquet'\n",
                "p_model = out_audit_dir / 'df_modelagem.parquet'\n",
                "\n",
                "if p_audit.exists(): shutil.rmtree(p_audit)\n",
                "if p_model.exists(): shutil.rmtree(p_model)\n",
                "\n",
                "df_target_audit.write.mode('overwrite').parquet(str(p_audit))\n",
                "df_modelagem.write.mode('overwrite').parquet(str(p_model))\n",
                "\n",
                "# Validação de releitura\n",
                "df_audit_chk = spark.read.parquet(str(p_audit))\n",
                "df_model_chk = spark.read.parquet(str(p_model))\n",
                "\n",
                "print(f\"Validação de Leitura de Parquet:\")\n",
                "print(f\"  - df_target_audit.parquet: {df_audit_chk.count():,} registros (SUCESSO: {(p_audit / '_SUCCESS').exists()})\")\n",
                "print(f\"  - df_modelagem.parquet:    {df_model_chk.count():,} registros (SUCESSO: {(p_model / '_SUCCESS').exists()})\")\n"
            ]
        }
    ]

    # Substituir ou anexar as novas células de escopo Nordeste
    # Procurar se já existe seção 8 e substituir/anexar
    cells = nb["cells"]
    
    # Encontrar onde inserir a nova seção 8 do Nordeste (após a seção 7)
    insert_idx = None
    for idx, c in enumerate(cells):
        if c["cell_type"] == "markdown" and c["source"] and "## 7. Limitação do Target Individual" in c["source"][0]:
            insert_idx = idx + 1
            break
            
    if insert_idx is None:
        # Se não achou a seção 7, insere antes da 8 antiga ou no final
        insert_idx = len(cells)

    print(f"Inserindo seção '8. Adequação do Escopo — Região Nordeste' no índice de célula {insert_idx}...")
    nb["cells"] = cells[:insert_idx] + nordeste_cells + cells[insert_idx:]

    print(f"Gravando notebook atualizado em: {nb_path}")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Notebook atualizado com sucesso!")

if __name__ == "__main__":
    main()
