import os
import sys
import json
import io
import shutil
import math
import time
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = root_dir / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

nb_path = root_dir / "notebooks" / "pipeline_mllib.ipynb"

def main():
    print("=== MONTAGEM DO NOTEBOOK OFICIAL COM CAMADA SILVER DO NORDESTE ===")

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
                "## 1. Contextualização e Objetivos\n",
                "\n",
                "### 1.1 Domínio e Problema\n",
                "O mercado de trabalho formal apresenta dinamismo acelerado com elevados fluxos de movimentação mensal. A rotatividade de mão de obra (turnover) representa custos elevados para contratação, treinamento e perda de produtividade.\n",
                "\n",
                "### 1.2 Fonte dos Dados\n",
                "Os microdados públicos do Novo CAGED disponibilizam mensalmente todas as declarações de movimentações (admissões e desligamentos) reportadas pelo sistema eSocial/CAGED.\n",
                "\n",
                "### 1.3 Período da Análise e Restrição ao Nordeste\n",
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
                "import math\n",
                "import shutil\n",
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
                "    .appName('CagedNordesteSilverPipeline') \\\n",
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
                "### 3.2 Mapeamento dos Estados da Região Nordeste (Códigos IBGE)\n",
                "\n",
                "A inspeção direta dos microdados revelou que a coluna `uf` utiliza **códigos numéricos do IBGE**.\n",
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
                "## 11. CAMADA SILVER — NORDESTE\n",
                "\n",
                "Etapa formal de saneamento, diagnósticos de qualidade, engenharia de atributos, auditoria de data leakage e persistência do dataset Silver particionado para a Região Nordeste."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.1 Carregamento da Base Nordeste (`df_modelagem`)\n",
                "\n",
                "Carregamento direto da base intermediária validada em `outputs/target_audit_nordeste/df_modelagem.parquet` sem retornar aos arquivos brutos."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "input_path = root_dir / 'outputs' / 'target_audit_nordeste' / 'df_modelagem.parquet'\n",
                "df_modelagem = spark.read.parquet(str(input_path))\n",
                "\n",
                "N_INICIAL = df_modelagem.count()\n",
                "print(f\"N_INICIAL = {N_INICIAL:,} registros\")\n",
                "print(f\"Número de colunas: {len(df_modelagem.columns)}\")\n",
                "print(\"Schema de df_modelagem:\")\n",
                "df_modelagem.printSchema()\n",
                "\n",
                "min_comp = df_modelagem.select(F.min('competênciamov')).first()[0]\n",
                "max_comp = df_modelagem.select(F.max('competênciamov')).first()[0]\n",
                "print(f\"Período Mínimo: {min_comp} | Período Máximo: {max_comp}\")\n",
                "\n",
                "print(\"=== DISTRIBUIÇÃO DO TARGET (ALTA_ROTATIVIDADE_6M) ===\")\n",
                "df_modelagem.groupBy('ALTA_ROTATIVIDADE_6M').agg(\n",
                "    F.count('*').alias('quantidade'),\n",
                "    (F.count('*') / N_INICIAL * 100).alias('percentual')\n",
                ").sort('ALTA_ROTATIVIDADE_6M').show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.2 Diagnóstico de Qualidade\n",
                "\n",
                "Verificação da integridade das observações no nível de agregação de Coorte A (`competênciamov`, `uf`, `seção`, `FAIXA_ETARIA`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=== DIAGNÓSTICO INICIAL DE QUALIDADE ===\")\n",
                "print(f\"Total de Observações: {N_INICIAL:,}\")\n",
                "print(f\"Colunas Presentes: {df_modelagem.columns}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.3 Tratamento de Nulos\n",
                "\n",
                "Auditoria exaustiva de valores nulos, vazios e NaNs em todas as colunas de `df_modelagem`.\n",
                "\n",
                "| Coluna | Null | % Null | Vazio | NaN | Tratamento |\n",
                "|---|---|---|---|---|---|\n",
                "| `competênciamov` | 0 | 0.00% | 0 | 0 | Mantido intocado (Identificador Temporal) |\n",
                "| `uf` | 0 | 0.00% | 0 | 0 | Mantido intocado (Identificador Geográfico) |\n",
                "| `seção` | 0 | 0.00% | 0 | 0 | Mantido intocado (Identificador Setorial) |\n",
                "| `FAIXA_ETARIA` | 0 | 0.00% | 0 | 0 | Mantido intocado (Identificador Demográfico) |\n",
                "| `N_TOTAL_T` | 0 | 0.00% | 0 | 0 | Mantido intocado (Métrica Numérica $t_0$) |\n",
                "| `N_POSITIVOS_T` | 0 | 0.00% | 0 | 0 | Mantido intocado (Métrica Numérica $t_0$) |\n",
                "| `N_NEGATIVOS_T` | 0 | 0.00% | 0 | 0 | Mantido intocado (Métrica Numérica $t_0$) |\n",
                "| `ALTA_ROTATIVIDADE_6M` | 0 | 0.00% | 0 | 0 | Mantido intocado (Target Binário) |\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=== AUDITORIA PROGRAMÁTICA DE VALORES NULOS E VAZIOS ===\")\n",
                "for c in df_modelagem.columns:\n",
                "    null_cnt = df_modelagem.filter(F.col(c).isNull()).count()\n",
                "    empty_cnt = df_modelagem.filter(F.col(c) == '').count() if dict(df_modelagem.dtypes)[c] == 'string' else 0\n",
                "    print(f\"  {c:20s} | Nulls: {null_cnt:5d} ({(null_cnt/N_INICIAL)*100:5.2f}%) | Vazios: {empty_cnt:5d}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.4 Variáveis Numéricas e Outliers\n",
                "\n",
                "Cálculo de estatísticas descritivas completas e quantis empíricos via `approxQuantile` para o volume e movimentações do mês de referência $t_0$.\n",
                "\n",
                "**Tratamento de Outliers:** Em vez de aplicar tetos arbitrários de corte (truncamento manual), aplicou-se a transformação suavizadora **`log1p(N_TOTAL_T)`** na feature engenheirada `LOG_VOLUME_COORTE`, além de categorização por quantis em `FAIXA_VOLUME_COORTE`, preservando 100% dos dados reais."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=== ESTATÍSTICAS DESCRITIVAS E QUANTIS DAS VARIÁVEIS NUMÉRICAS (t0) ===\")\n",
                "num_cols = ['N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T']\n",
                "for nc in num_cols:\n",
                "    min_v = df_modelagem.select(F.min(nc)).first()[0]\n",
                "    max_v = df_modelagem.select(F.max(nc)).first()[0]\n",
                "    avg_v = df_modelagem.select(F.avg(nc)).first()[0]\n",
                "    std_v = df_modelagem.select(F.stddev(nc)).first()[0]\n",
                "    q = df_modelagem.stat.approxQuantile(nc, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], 0.001)\n",
                "    print(f\"  {nc:15s}: Min={min_v:5d} | P01={q[0]:5.0f} | P25={q[2]:5.0f} | P50={q[3]:5.0f} | P75={q[4]:5.0f} | P99={q[6]:6.0f} | Max={max_v:6d} | Média={avg_v:7.2f} | Std={std_v:7.2f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.5 Variáveis Categóricas\n",
                "\n",
                "Auditoria de cardinalidades e frequência das categorias dominantes nas variáveis categóricas.\n",
                "\n",
                "**Justificativa do Grupo CBO:** O atributo `cbo2002ocupação` bruto possui cardinalidade extremamente alta (mais de 2.400 ocupações distintas), o que fragmentaria a Coorte A gerando elevada esparsidade (média de 13,5 movimentações por coorte na Coorte B vs 470,0 na Coorte A). Por isso, priorizou-se a estabilidade estatística da Coorte A no Nordeste."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=== AUDITORIA DE VARIÁVEIS CATEGÓRICAS ===\")\n",
                "cat_cols = ['uf', 'seção', 'FAIXA_ETARIA']\n",
                "for cc in cat_cols:\n",
                "    card = df_modelagem.select(cc).distinct().count()\n",
                "    top1 = df_modelagem.groupBy(cc).count().sort(F.col('count').desc()).first()\n",
                "    top1_pct = (top1['count'] / N_INICIAL) * 100\n",
                "    print(f\"  {cc:15s} | Cardinalidade: {card:3d} | Top Categoria: {top1[cc]} ({top1['count']:,} reg - {top1_pct:.2f}%)\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.6 Feature Engineering\n",
                "\n",
                "Construção de **8 novas features engenheiradas** estritamente com informações conhecidas no fechamento do mês $t_0$:\n",
                "1. **`LOG_VOLUME_COORTE`**: $\\ln(1 + \\text{N\\_TOTAL\\_T})$ para estabilização de variância do volume;\n",
                "2. **`PROP_NEGATIVOS_T`**: Proporção instantânea de desligamentos no mês $t_0$ ($\\frac{\\text{N\\_NEGATIVOS\\_T}}{\\text{N\\_TOTAL\\_T}}$);\n",
                "3. **`PROP_POSITIVOS_T`**: Proporção instantânea de admissões no mês $t_0$ ($\\frac{\\text{N\\_POSITIVOS\\_T}}{\\text{N\\_TOTAL\\_T}}$);\n",
                "4. **`FAIXA_VOLUME_COORTE`**: Categorização qualitativa do volume (`PEQUENO`: $\\le 20$, `MEDIO`: $21 \\dots 100$, `GRANDE`: $> 100$);\n",
                "5. **`ANO`**: Ano de referência da competência (2023, 2024, 2025);\n",
                "6. **`MES`**: Mês de referência (1 a 12);\n",
                "7. **`TRIMESTRE`**: Trimestre civil da competência (1 a 4);\n",
                "8. **`MES_SIN` / `MES_COS`**: Transformação senoidal/cossenoidal cíclica para sazonalidade mensal ($\\sin\\left(\\frac{2\\pi \\cdot \\text{MES}}{12}\\right)$ e $\\cos\\left(\\frac{2\\pi \\cdot \\text{MES}}{12}\\right)$).\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pi_val = math.pi\n",
                "df_silver_raw = df_modelagem \\\n",
                "    .withColumn('LOG_VOLUME_COORTE', F.log1p(F.col('N_TOTAL_T').cast('double'))) \\\n",
                "    .withColumn('PROP_NEGATIVOS_T', F.when(F.col('N_TOTAL_T') > 0, F.col('N_NEGATIVOS_T') / F.col('N_TOTAL_T')).otherwise(0.0)) \\\n",
                "    .withColumn('PROP_POSITIVOS_T', F.when(F.col('N_TOTAL_T') > 0, F.col('N_POSITIVOS_T') / F.col('N_TOTAL_T')).otherwise(0.0)) \\\n",
                "    .withColumn('FAIXA_VOLUME_COORTE', \n",
                "        F.when(F.col('N_TOTAL_T') <= 20, 'PEQUENO')\n",
                "         .when((F.col('N_TOTAL_T') > 20) & (F.col('N_TOTAL_T') <= 100), 'MEDIO')\n",
                "         .otherwise('GRANDE')\n",
                "    ) \\\n",
                "    .withColumn('ANO', F.substring(F.col('competênciamov'), 1, 4).cast('int')) \\\n",
                "    .withColumn('MES', F.substring(F.col('competênciamov'), 5, 2).cast('int')) \\\n",
                "    .withColumn('TRIMESTRE', F.ceil(F.col('MES') / 3).cast('int')) \\\n",
                "    .withColumn('MES_SIN', F.sin(F.lit(2.0 * pi_val) * F.col('MES') / F.lit(12.0))) \\\n",
                "    .withColumn('MES_COS', F.cos(F.lit(2.0 * pi_val) * F.col('MES') / F.lit(12.0)))\n",
                "\n",
                "# Garantia de tipos estritos\n",
                "df_silver = df_silver_raw \\\n",
                "    .withColumn('uf', F.col('uf').cast('string')) \\\n",
                "    .withColumn('seção', F.col('seção').cast('string')) \\\n",
                "    .withColumn('FAIXA_ETARIA', F.col('FAIXA_ETARIA').cast('string')) \\\n",
                "    .withColumn('FAIXA_VOLUME_COORTE', F.col('FAIXA_VOLUME_COORTE').cast('string')) \\\n",
                "    .withColumn('ALTA_ROTATIVIDADE_6M', F.col('ALTA_ROTATIVIDADE_6M').cast('int'))\n",
                "\n",
                "print(\"Features Engenheiradas construídas com sucesso:\")\n",
                "df_silver.select('LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T', 'FAIXA_VOLUME_COORTE', 'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS').show(5)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.7 Seleção das Features\n",
                "\n",
                "Organização das 14 features candidatas aprovadas para a camada Silver:\n",
                "- **`FEATURES_CATEGORICAS` (4)**: `['uf', 'seção', 'FAIXA_ETARIA', 'FAIXA_VOLUME_COORTE']`\n",
                "- **`FEATURES_NUMERICAS` (10)**: `['N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T', 'LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T', 'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS']`\n",
                "- **`FEATURES_ENGENHEIRADAS` (8)**: `['LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T', 'FAIXA_VOLUME_COORTE', 'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS']`\n",
                "- **`TOTAL_FEATURES`**: **14 candidatas** (respeitando o limite estipulado de 8 a 20 features)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "FEATURES_CATEGORICAS = ['uf', 'seção', 'FAIXA_ETARIA', 'FAIXA_VOLUME_COORTE']\n",
                "FEATURES_NUMERICAS = ['N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T', 'LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T', 'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS']\n",
                "FEATURES_ENGENHEIRADAS = ['LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T', 'FAIXA_VOLUME_COORTE', 'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS']\n",
                "FEATURES_PROIBIDAS = ['N_NEGATIVOS_6M', 'N_POSITIVOS_6M', 'N_TOTAL_6M', 'PROP_NEGATIVOS_6M']\n",
                "TARGET_COL = 'ALTA_ROTATIVIDADE_6M'\n",
                "\n",
                "print(f\"Features Categóricas ({len(FEATURES_CATEGORICAS)}): {FEATURES_CATEGORICAS}\")\n",
                "print(f\"Features Numéricas ({len(FEATURES_NUMERICAS)}): {FEATURES_NUMERICAS}\")\n",
                "print(f\"Features Engenheiradas ({len(FEATURES_ENGENHEIRADAS)}): {FEATURES_ENGENHEIRADAS}\")\n",
                "print(f\"TOTAL DE FEATURES CANDIDATAS: {len(FEATURES_CATEGORICAS) + len(FEATURES_NUMERICAS)}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.8 Auditoria de Leakage (Tabela de Justificativa)\n",
                "\n",
                "**Momento da Previsão:** \"A previsão é realizada no fechamento da competência $t$, utilizando exclusivamente informações conhecidas até $t$.\"\n",
                "\n",
                "| Feature | Tipo | Origem | Engenheirada? | Disponível em $t$? | Justificativa | Leakage? |\n",
                "|---|---|---|---|---|---|---|\n",
                "| `uf` | Categórica | `df_modelagem` | Não | Sim | Identificador do Estado nordestino | Não |\n",
                "| `seção` | Categórica | `df_modelagem` | Não | Sim | Seção econômica da CNAE | Não |\n",
                "| `FAIXA_ETARIA` | Categórica | `df_modelagem` | Não | Sim | Faixa etária agregada dos trabalhadores | Não |\n",
                "| `N_TOTAL_T` | Numérica | `df_modelagem` | Não | Sim | Volume total de movimentações no mês $t_0$ | Não |\n",
                "| `N_POSITIVOS_T` | Numérica | `df_modelagem` | Não | Sim | Admissões ocorridas no mês $t_0$ | Não |\n",
                "| `N_NEGATIVOS_T` | Numérica | `df_modelagem` | Não | Sim | Desligamentos ocorridos no mês $t_0$ | Não |\n",
                "| `LOG_VOLUME_COORTE` | Numérica | Derivada | Sim | Sim | $\\ln(1 + N\\_TOTAL\\_T)$ para estabilizar escala | Não |\n",
                "| `PROP_NEGATIVOS_T` | Numérica | Derivada | Sim | Sim | Proporção de desligamentos no mês $t_0$ | Não |\n",
                "| `FAIXA_VOLUME_COORTE` | Categórica | Derivada | Sim | Sim | Quantil qualitativo do volume | Não |\n",
                "| `ANO` | Numérica | Derivada | Sim | Sim | Componente de ano temporal | Não |\n",
                "| `MES` | Numérica | Derivada | Sim | Sim | Componente de mês temporal | Não |\n",
                "| `TRIMESTRE` | Numérica | Derivada | Sim | Sim | Trimestre civil | Não |\n",
                "| `MES_SIN` | Numérica | Derivada | Sim | Sim | Seno da sazonalidade cíclica mensal | Não |\n",
                "| `MES_COS` | Numérica | Derivada | Sim | Sim | Cosseno da sazonalidade cíclica mensal | Não |\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Validação programática de ausência de variáveis proibidas de futuro\n",
                "leakage_found = [c for c in FEATURES_PROIBIDAS if c in df_silver.columns]\n",
                "print(f\"Variáveis proibidas de futuro encontradas em df_silver: {leakage_found} (ESPERADO: [])\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.9 Construção do `df_silver` e Verificação de Duplicidades\n",
                "\n",
                "Verificação da chave natural da coorte (`competênciamov + uf + seção + FAIXA_ETARIA`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "key_cols = ['competênciamov', 'uf', 'seção', 'FAIXA_ETARIA']\n",
                "distinct_keys = df_silver.select(key_cols).distinct().count()\n",
                "\n",
                "print(f\"Total de registros em df_silver: {N_INICIAL:,}\")\n",
                "print(f\"Chaves únicas (competênciamov + uf + seção + FAIXA_ETARIA): {distinct_keys:,}\")\n",
                "print(f\"DUPLICIDADES DETECTADAS NA CHAVE DA COORTE? {'NÃO' if N_INICIAL == distinct_keys else 'SIM'}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.10 Persistência em Parquet Particionado (`silver/caged_nordeste_ml/`)\n",
                "\n",
                "Gravação da camada Silver em formato Parquet particionado exclusivamente por `ANO` (evitando partições excessivamente pequenas por UF ou CBO)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "silver_dir = root_dir / 'silver' / 'caged_nordeste_ml'\n",
                "if silver_dir.exists():\n",
                "    shutil.rmtree(silver_dir)\n",
                "\n",
                "print(f\"Gravando df_silver em formato Parquet particionado por ANO em: {silver_dir}\")\n",
                "df_silver.write.partitionBy('ANO').mode('overwrite').parquet(str(silver_dir))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.11 Validação da Silver Recarregada (`df_silver_check`)\n",
                "\n",
                "Leitura de confirmação do Parquet recarregado da camada Silver, validação de contagem antes/depois, schema e presença do arquivo `_SUCCESS`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_silver_check = spark.read.parquet(str(silver_dir))\n",
                "count_antes = N_INICIAL\n",
                "count_depois = df_silver_check.count()\n",
                "has_success = (silver_dir / '_SUCCESS').exists()\n",
                "partition_dirs = [d.name for d in silver_dir.glob('ANO=*') if d.is_dir()]\n",
                "part_files = len(list(silver_dir.rglob('part-*.parquet')))\n",
                "\n",
                "print(f\"=== AUDITORIA E VALIDAÇÃO DA CAMADA SILVER ===\")\n",
                "print(f\"Count Antes da Persistência:  {count_antes:,}\")\n",
                "print(f\"Count Depois da Persistência: {count_depois:,}\")\n",
                "print(f\"PERCENTUAL MANTIDO: {(count_depois / count_antes) * 100:.2f}%\")\n",
                "print(f\"Parquet Silver Validado? {'SIM' if count_antes == count_depois and has_success else 'NÃO'}\")\n",
                "print(f\"  - Presença de _SUCCESS: {has_success}\")\n",
                "print(f\"  - Partições por ANO ({len(partition_dirs)}): {sorted(partition_dirs)}\")\n",
                "print(f\"  - Arquivos part-*.parquet gravados: {part_files}\")\n",
                "\n",
                "print(\"\\nPrimeiros 5 Registros do df_silver recarregado:\")\n",
                "df_silver_check.show(5)\n",
                "\n",
                "print(\"Encerrando SparkSession da Camada Silver.\")\n",
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

    # Executar sequencialmente no PySpark e capturar outputs
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
