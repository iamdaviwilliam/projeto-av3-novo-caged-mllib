import os
import sys
import json
import io
import shutil
import time
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = root_dir / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

nb_path = root_dir / "notebooks" / "pipeline_mllib.ipynb"
summary_path = root_dir / "outputs" / "nordeste_summary.json"

def main():
    print("=== MONTAGEM E EXECUÇÃO DO NOTEBOOK OFICIAL DO NORDESTE ===")
    
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Projeto MLlib — Previsão de Alta Rotatividade Agregada no Novo CAGED (2023–2025)\n",
                "\n",
                "- **Disciplina:** Big Data e Processamento Distribuído\n",
                "- **Domínio:** Mercado de Trabalho brasileiro\n",
                "- **Fonte de Dados:** Microdados do Novo CAGED (Ministério do Trabalho e Emprego - MTE)\n",
                "- **Período Observado:** 2023 a 2025 (35 competências mensais extraídas)\n",
                "- **Escopo de Análise e Modelagem:** **Região Nordeste (AL, BA, CE, MA, PB, PE, PI, RN, SE)**\n",
                "- **Tecnologia Obrigatória:** PySpark (`pyspark.sql`)\n",
                "- **Abordagem Metodológica:** Análise Agregada por Coortes / Perfis Socioeconômicos\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Contextualização\n",
                "\n",
                "### 1.1 Domínio e Problema\n",
                "O mercado de trabalho formal apresenta dinamismo acelerado com elevados fluxos de movimentação mensal. A rotatividade de mão de obra (turnover) representa custos elevados para contratação, treinamento e perda de produtividade.\n",
                "\n",
                "### 1.2 Fonte dos Dados\n",
                "Os microdados públicos do Novo CAGED disponibilizam mensalmente todas as declarações de movimentações (admissões e desligamentos) reportadas pelo sistema eSocial/CAGED.\n",
                "\n",
                "### 1.3 Período da Análise e Restrição do Escopo Geográfico\n",
                "O estudo engloba o triênio **2023 a 2025** (35 competências). Para viabilidade computacional em ambiente local com 8 GB de RAM, o escopo foi focado nos 9 estados da **Região Nordeste**, acumulando **18.996.006 registros de movimentação**.\n",
                "\n",
                "### 1.4 Objetivo da Aprendizagem de Máquina\n",
                "Identificar perfis socioeconômicos e setoriais com alta propensão a apresentar **alta taxa de movimentações negativas nos 6 meses subsequentes**."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Configuração Conservadora do Ambiente PySpark\n",
                "\n",
                "Inicialização da `SparkSession` com parâmetros conservadores ajustados para máquina local com 8 GB de memória RAM:\n",
                "- `master('local[2]')` (evitando `local[*]`);\n",
                "- `spark.driver.memory` = `3g`;\n",
                "- `spark.sql.shuffle.partitions` = `32`;\n",
                "- `spark.sql.adaptive.enabled` = `true`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "from pyspark.sql import SparkSession\n",
                "import pyspark.sql.functions as F\n",
                "\n",
                "root_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "hadoop_home = (root_dir / 'hadoop').resolve()\n",
                "os.environ['HADOOP_HOME'] = str(hadoop_home)\n",
                "os.environ['PATH'] = str(hadoop_home / 'bin') + os.pathsep + os.environ.get('PATH', '')\n",
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
                "## 3. Adequação do Escopo — Região Nordeste\n",
                "\n",
                "### 3.1 Diagnóstico de Estabilidade Local e Decisão Metodológica\n",
                "- **Estabilidade Operacional em 8 GB RAM:** A análise inicial da base bruta nacional para o triênio 2023–2025 ultrapassou 118 milhões de movimentações. O processamento integral dessa base em máquina local com 8 GB de RAM apresentou instabilidade e estouro de memória no JVM/driver.\n",
                "- **Restrição Estratégica ao Nordeste:** A análise final foi adaptada para considerar exclusivamente os 9 estados nordestinos (AL, BA, CE, MA, PB, PE, PI, RN, SE).\n",
                "- **Filtragem Precoce (Na Leitura Mensal):** A filtragem dos estados passou a ocorrer **logo após a leitura de cada arquivo mensal de forma individual**, seguida de seleção de colunas e escrita imediata na camada intermediária Parquet (`outputs/nordeste_monthly/`). Isso evitou qualquer operação pesada `union` em memória sobre o dataset nacional.\n",
                "- **Manutenção Integral da Metodologia:** O período histórico (2023–2025), a tecnologia PySpark, a definição de coortes, a censura à direita e o cálculo de janelas futuras foram reconstruídos especificamente para o universo do Nordeste.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.2 Representação Real da Coluna `uf` (Códigos IBGE)\n",
                "\n",
                "A inspeção direta do cabeçalho e valores dos microdados revelou que a coluna `uf` não utiliza siglas (como 'BA' ou 'PE'), mas sim **códigos numéricos do IBGE**.\n",
                "\n",
                "Mapeamento explícito dos nove estados da Região Nordeste:\n",
                "- `21`: Maranhão (MA)\n",
                "- `22`: Piauí (PI)\n",
                "- `23`: Ceará (CE)\n",
                "- `24`: Rio Grande do Norte (RN)\n",
                "- `25`: Paraíba (PB)\n",
                "- `26`: Pernambuco (PE)\n",
                "- `27`: Alagoas (AL)\n",
                "- `28`: Sergipe (SE)\n",
                "- `29`: Bahia (BA)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Definição dos códigos IBGE dos 9 estados do Nordeste\n",
                "UFS_NORDESTE = [21, 22, 23, 24, 25, 26, 27, 28, 29]\n",
                "UF_MAP = {\n",
                "    21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB',\n",
                "    26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA'\n",
                "}\n",
                "\n",
                "print(f\"UFS_NORDESTE = {UFS_NORDESTE}\")\n",
                "print(f\"Mapeamento IBGE -> UF: {UF_MAP}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.3 Leitura e Validação da Camada Intermediária do Nordeste (`outputs/nordeste_monthly/`)\n",
                "\n",
                "Confirmação dos arquivos mensais persistidos em Parquet (organizados por `competencia=YYYYMM/`) contendo os arquivos `part-*.parquet` e `_SUCCESS`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "monthly_path = root_dir / 'outputs' / 'nordeste_monthly'\n",
                "df_ne_raw = spark.read.parquet(str(monthly_path / 'competencia=*'))\n",
                "total_nordeste_recs = df_ne_raw.count()\n",
                "print(f\"Caminho dos Parquets: {monthly_path}\")\n",
                "print(f\"Total de movimentações do Nordeste (2023–2025): {total_nordeste_recs:,} registros\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.4 Contagem de Movimentações por Estado e Ano e Auditoria de Competências\n",
                "\n",
                "Verificação da distribuição das movimentações por UF e Ano, e verificação das competências temporais presentes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Contagem por Estado e Ano\n",
                "df_with_year = df_ne_raw.withColumn('ano', F.substring(F.col('competênciamov'), 1, 4).cast('int'))\n",
                "uf_counts_df = df_with_year.groupBy('uf', 'ano').count().collect()\n",
                "\n",
                "uf_year_summary = {}\n",
                "for row in uf_counts_df:\n",
                "    uf_code = row['uf']\n",
                "    uf_sigla = UF_MAP.get(uf_code, str(uf_code))\n",
                "    ano = row['ano']\n",
                "    cnt = row['count']\n",
                "    uf_year_summary.setdefault(uf_sigla, {})[ano] = cnt\n",
                "\n",
                "print(\"=== CONTAGEM DE MOVIMENTAÇÕES POR ESTADO E ANO (NORDESTE) ===\")\n",
                "for sigla in sorted(UF_MAP.values()):\n",
                "    c2023 = uf_year_summary.get(sigla, {}).get(2023, 0)\n",
                "    c2024 = uf_year_summary.get(sigla, {}).get(2024, 0)\n",
                "    c2025 = uf_year_summary.get(sigla, {}).get(2025, 0)\n",
                "    tot = c2023 + c2024 + c2025\n",
                "    print(f\"  {sigla}: 2023 = {c2023:9,} | 2024 = {c2024:9,} | 2025 = {c2025:9,} | TOTAL = {tot:10,}\")\n",
                "\n",
                "total_2023 = sum(uf_year_summary.get(s, {}).get(2023, 0) for s in UF_MAP.values())\n",
                "total_2024 = sum(uf_year_summary.get(s, {}).get(2024, 0) for s in UF_MAP.values())\n",
                "total_2025 = sum(uf_year_summary.get(s, {}).get(2025, 0) for s in UF_MAP.values())\n",
                "total_nordeste = total_2023 + total_2024 + total_2025\n",
                "\n",
                "print(f\"\\nTOTAL NORDESTE 2023: {total_2023:,}\")\n",
                "print(f\"TOTAL NORDESTE 2024: {total_2024:,}\")\n",
                "print(f\"TOTAL NORDESTE 2025: {total_2025:,}\")\n",
                "print(f\"TOTAL NORDESTE ACUMULADO: {total_nordeste:,}\")\n",
                "print(f\"CRITÉRIO MÍNIMO DE 500 MIL REGISTROS ATENDIDO? {'SIM' if total_nordeste >= 500000 else 'NÃO'}\")\n",
                "\n",
                "# Auditoria de Competências\n",
                "distinct_comps = sorted([r['competênciamov'] for r in df_ne_raw.select('competênciamov').distinct().collect()])\n",
                "all_possible_comps = [f'2023{m:02d}' for m in range(1, 13)] + [f'2024{m:02d}' for m in range(1, 13)] + [f'2025{m:02d}' for m in range(1, 13)]\n",
                "missing_comps = [c for c in all_possible_comps if c not in distinct_comps]\n",
                "\n",
                "print(f\"\\nCompetências Processadas ({len(distinct_comps)}): {distinct_comps}\")\n",
                "print(f\"Competências Ausentes ({len(missing_comps)}): {missing_comps}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Reconstrução e Comparação de Coortes no Universo Nordeste\n",
                "\n",
                "Reavaliação empírica das 3 granularidades de coorte (A, B e C) utilizando exclusivamente os dados da Região Nordeste."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Preparação das colunas de agrupamento socioeconômico\n",
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
                "def evaluate_cohort(df, cols, name):\n",
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
                "evaluate_cohort(df_prep, ['competênciamov', 'uf', 'seção', 'FAIXA_ETARIA'], 'Coorte A (Agregada: competencia + uf + seção + FAIXA_ETARIA)')\n",
                "evaluate_cohort(df_prep, ['competênciamov', 'uf', 'seção', 'GRUPO_CBO', 'FAIXA_ETARIA', 'graudeinstrução'], 'Coorte B (Intermediária)')\n",
                "evaluate_cohort(df_prep, ['competênciamov', 'uf', 'seção', 'GRUPO_CBO', 'FAIXA_ETARIA', 'graudeinstrução', 'sexo', 'FAIXA_SALARIAL'], 'Coorte C (Detalhada)')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4.1 Reavaliação de Limiares Mínimos de Tamanho de Coorte no Nordeste\n",
                "\n",
                "Comparação explícita de limiares mínimos (10, 20, 50, 100) para a Coorte A selecionada."
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
                "print(\"=== SIMULAÇÃO DE LIMIARES MÍNIMOS DE TAMANHO DE COORTE ===\")\n",
                "for thresh in [10, 20, 50, 100]:\n",
                "    filt = grouped_a.filter(F.col('n_total') >= thresh)\n",
                "    rem_c = filt.count()\n",
                "    pct_c_rem = ((total_coortes_a - rem_c) / total_coortes_a) * 100\n",
                "    recs_rep = filt.select(F.sum('n_total')).first()[0]\n",
                "    recs_rep = recs_rep if recs_rep is not None else 0\n",
                "    pct_recs = (recs_rep / total_recs_brutos) * 100\n",
                "    print(f\"Limiar mínimo {thresh:3d} registros/coorte:\")\n",
                "    print(f\"  - Coortes mantidas: {rem_c:7,} | Removidas: {pct_c_rem:6.2f}%\")\n",
                "    print(f\"  - Registros preservados: {recs_rep:11,} ({pct_recs:5.2f}% do volume Nordeste)\\n\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Reconstrução do Target Agregado e Janela Futura no Nordeste\n",
                "\n",
                "Construção do indicador `PROP_NEGATIVOS_6M` observando o mesmo perfil agregado ($uf \\times seção \\times FAIXA\\_ETARIA$) nos 6 meses futuros ($t+1 \\dots t+6$).\n",
                "\n",
                "### 5.1 Censura à Direita e Elegibilidade de Referência ($t_0 \\le 202506$)\n",
                "Como os dados terminam em 202512, a última competência de referência elegível que possui 6 meses futuros completos é **202506**."
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
                "# Associação de 6 meses futuros (Join sem produto cartesiano)\n",
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
                "print(f\"Quantis de PROP_NEGATIVOS_6M no Nordeste:\")\n",
                "print(f\"  P25_NORDESTE: {quantiles_ne[0]:.6f}\")\n",
                "print(f\"  P50_NORDESTE (Mediana): {p50_nordeste:.6f}\")\n",
                "print(f\"  P75_NORDESTE: {p75_nordeste:.6f}\")\n",
                "\n",
                "df_target_audit = df_target_audit \\\n",
                "    .withColumn('ALTA_ROTATIVIDADE_6M', F.when(F.col('PROP_NEGATIVOS_6M') > p50_nordeste, 1).otherwise(0))\n",
                "\n",
                "print(\"\\n=== DISTRIBUIÇÃO DAS CLASSES DO TARGET BINÁRIO (ALTA_ROTATIVIDADE_6M) ===\")\n",
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
                "## 6. Auditoria de Data Leakage e Separação de DataFrames Lógicos\n",
                "\n",
                "Separação estrita entre `df_target_audit` (com histórico futuro completo para validação metodológica) e `df_modelagem` (contendo apenas features do mês de referência $t_0$ e o target binário `ALTA_ROTATIVIDADE_6M`).\n",
                "\n",
                "Verificação formal de ausência de vazamento de dados em `df_modelagem`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Construção do DataFrame Lógico de Modelagem (Sem variáveis futuras)\n",
                "df_modelagem = df_target_audit.select(\n",
                "    'competênciamov', 'uf', 'seção', 'FAIXA_ETARIA',\n",
                "    'N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T',\n",
                "    'ALTA_ROTATIVIDADE_6M'\n",
                ")\n",
                "\n",
                "forbidden_cols = ['N_NEGATIVOS_6M', 'N_POSITIVOS_6M', 'N_TOTAL_6M', 'PROP_NEGATIVOS_6M']\n",
                "leakage_check = [c for c in forbidden_cols if c in df_modelagem.columns]\n",
                "print(f\"Variáveis futuras encontradas em df_modelagem: {leakage_check} (ESPERADO: [])\")\n",
                "print(f\"Validação de Target Nulo: {df_modelagem.filter(F.col('ALTA_ROTATIVIDADE_6M').isNull()).count()} nulos em ALTA_ROTATIVIDADE_6M\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Persistência dos DataFrames Intermediários em Parquet (`outputs/target_audit_nordeste/`)\n",
                "\n",
                "Gravação física dos Parquets intermediários do Nordeste e validação de releitura."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "out_audit_dir = root_dir / 'outputs' / 'target_audit_nordeste'\n",
                "out_audit_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "path_audit_parquet = out_audit_dir / 'df_target_audit.parquet'\n",
                "path_model_parquet = out_audit_dir / 'df_modelagem.parquet'\n",
                "\n",
                "if path_audit_parquet.exists(): shutil.rmtree(path_audit_parquet)\n",
                "if path_model_parquet.exists(): shutil.rmtree(path_model_parquet)\n",
                "\n",
                "df_target_audit.write.mode('overwrite').parquet(str(path_audit_parquet))\n",
                "df_modelagem.write.mode('overwrite').parquet(str(path_model_parquet))\n",
                "\n",
                "# Validação de releitura dos arquivos gravados\n",
                "df_audit_read = spark.read.parquet(str(path_audit_parquet))\n",
                "df_model_read = spark.read.parquet(str(path_model_parquet))\n",
                "\n",
                "print(f\"Validação de Leitura de Parquet:\")\n",
                "print(f\"  - df_target_audit.parquet: {df_audit_read.count():,} registros | _SUCCESS: {(path_audit_parquet / '_SUCCESS').exists()}\")\n",
                "print(f\"  - df_modelagem.parquet:    {df_model_read.count():,} registros | _SUCCESS: {(path_model_parquet / '_SUCCESS').exists()}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Finalização do Pipeline Intermediário do Nordeste\n",
                "\n",
                "A base do Nordeste está devidamente limpa, estabilizada, com coortes reavaliadas, target reconstruído ($P_{50} = 0,479005$) e intermediários persistidos fisicamente.\n",
                "\n",
                "**Pausa para revisão:** As etapas de construção da camada Silver e treinamento de estimadores de Aprendizagem de Máquina (Regressão Logística, RandomForest, etc.) serão executadas na fase posterior do projeto."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Pipeline intermediário do Nordeste concluído com sucesso e gravado em disco!\")\n",
                "spark.stop()\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Executar sequencialmente as células no PySpark e salvar o notebook com outputs salvos
    exec_globals = {}
    print("Executando células sequencialmente no PySpark para capturar outputs...")
    execution_count = 0
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            execution_count += 1
            cell["execution_count"] = execution_count
            code = "".join(cell["source"])
            
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            
            print(f"Executando Célula #{execution_count} (índice {idx})...")
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code, exec_globals)
                
                output_text = stdout_buf.getvalue()
                error_text = stderr_buf.getvalue()
                
                cell_outputs = []
                if output_text:
                    cell_outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": output_text.splitlines(keepends=True)
                    })
                if error_text:
                    clean_errs = [l for l in error_text.splitlines(keepends=True) if "WARN" not in l and "incubator" not in l]
                    if clean_errs:
                        cell_outputs.append({
                            "name": "stderr",
                            "output_type": "stream",
                            "text": clean_errs
                        })
                cell["outputs"] = cell_outputs
                print(f"  -> Célula #{execution_count} OK! Output lines: {len(output_text.splitlines())}")
            except Exception as e:
                print(f"  -> ERRO na Célula #{execution_count}: {e}")
                cell_outputs = [{
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "output_type": "error",
                    "traceback": [str(e)]
                }]
                cell["outputs"] = cell_outputs
                break

    print(f"Gravando notebook final em: {nb_path}")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Validação do notebook gravado:")
    with open(nb_path, "r", encoding="utf-8") as f:
        check_nb = json.load(f)
    code_cnt = sum(1 for c in check_nb["cells"] if c["cell_type"] == "code")
    out_cnt = sum(1 for c in check_nb["cells"] if c["cell_type"] == "code" and len(c.get("outputs", [])) > 0)
    print(f"SUCESSO: {out_cnt}/{code_cnt} células de código executadas e com outputs persistidos!")

if __name__ == "__main__":
    main()
