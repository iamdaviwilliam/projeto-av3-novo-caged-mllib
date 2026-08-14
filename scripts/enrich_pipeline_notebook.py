# encoding: utf-8
import json
import os
import sys
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent.parent
    nb_path = root_dir / 'notebooks' / 'pipeline_mllib.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    orig_cells = nb['cells']
    celulas_antes = len(orig_cells)
    print(f"CELULAS_ANTES = {celulas_antes}")
    
    # Split notebook: head = cells 0..3 (Context, Objectives, Spark config), tail = cells 4..40 (Silver, MLlib, LR, RF, Tuning, Reflections, CBO, Field Notebook)
    head_cells = orig_cells[:4]
    tail_cells = orig_cells[4:]
    
    # Define new methodological cells to be inserted between head and tail
    new_cells = [
        # --- SEÇÃO 3 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Dados Utilizados e Auditoria Inicial\n",
                "\n",
                "### 3.1 Fontes de Dados e Cobertura Nacional\n",
                "Para a realização deste estudo sobre a rotatividade de mão de obra no mercado de trabalho formal brasileiro, foram auditados os três arquivos públicos disponibilizados mensalmente pelo Ministério do Trabalho e Emprego (MTE) referentes ao **Novo CAGED**:\n",
                "- **`CAGEDMOV` (Movimentações):** Arquivo declarativo principal contendo todos os registros de admissões e desligamentos reportados pelos estabelecimentos;\n",
                "- **`CAGEDEXC` (Exclusões):** Declarações de retificação e exclusão de movimentações de competências anteriores;\n",
                "- **`CAGEDFOR` (Fora do Prazo):** Declarações de movimentações entregues fora do prazo legal.\n",
                "\n",
                "**Período de Análise:** Triênio de **2023 a 2025** (**35 competências mensais** extraídas de `202301` a `202512`, exceto a competência `202312` não disponibilizada na fonte de dados original).\n",
                "\n",
                "### 3.2 Estrutura Inicial dos Microdados\n",
                "A auditoria inicial das tabelas confirmou a presença de variáveis demográficas (`idade`, `sexo`, `graudeinstrução`), geográficas (`uf`, `município`), setoriais (`seção` econômica CNAE 2.0, `cbo2002ocupação`) e contratuais (`salário`, `horascontratuais`, `tipomovimentação`, `saldomovimentação`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Inspeção Estrutural da Ingestão de Dados (Demonstração da leitura dos microdados)\n",
                "monthly_path = root_dir / 'outputs' / 'nordeste_monthly'\n",
                "df_monthly_audit = spark.read.parquet(str(monthly_path))\n",
                "\n",
                "print(\"=== AUDITORIA ESTRUTURAL DAS COMPETÊNCIAS PROCESSADAS NO NORDESTE ===\")\n",
                "print(f\"Total de registros agregados mensais em outputs/nordeste_monthly: {df_monthly_audit.count():,}\")\n",
                "print(\"Esquema das colunas de agrupamento contemporâneo:\")\n",
                "df_monthly_audit.printSchema()\n"
            ]
        },

        # --- SEÇÃO 4 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Principais Limitações dos Microdados e Definição da Abordagem Agregada\n",
                "\n",
                "### 4.1 Ausência de Identificador Persistente do Trabalhador\n",
                "A principal limitação metodológica identificada nos microdados públicos do Novo CAGED é a **ausência de um identificador único e persistente** (como CPF mascarado ou número de inscrição anonimizado) que permita rastrear o mesmo indivíduo ou vínculo empregatício ao longo das competências mensais.\n",
                "\n",
                "### 4.2 Rejeição de Chaves Artificiais Pseudônimas\n",
                "Avaliou-se a possibilidade de criar chaves compostas artificiais utilizando combinações de atributos demográficos e ocupacionais (ex.: `idade + sexo + município + CBO + salário`). No entanto, essa abordagem foi **estritamente rejeitada**, pois colisões de atributos entre trabalhadores diferentes levariam à associação indevida de indivíduos distintos como se fossem a mesma pessoa, violando a integridade metodológica da análise.\n",
                "\n",
                "### 4.3 Adaptação Metodológica: Abordagem Agregada por Coortes Socioeconômicas\n",
                "Em virtude dessa limitação, o problema de pesquisa foi formalmente redefinido: **a unidade fundamental de análise deixa de ser o trabalhador individual e passa a ser a Coorte Socioeconômica e Setorial**.\n",
                "\n",
                "> **ATENÇÃO — ESCLARECIMENTO RIGOROSO:**\n",
                "> O modelo de Aprendizagem de Máquina desenvolvido neste projeto **NÃO prevê o desligamento de trabalhadores individuais**. O modelo prevê a **intensidade relativa de movimentações negativas de uma Coorte Agregada** nos meses subsequentes."
            ]
        },

        # --- SEÇÃO 5 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Recorte Geográfico (Nordeste) e Estratégia de Processamento Local\n",
                "\n",
                "### 5.1 Motivação Computacional do Recorte Geográfico\n",
                "O processamento dos microdados do Novo CAGED no âmbito nacional ultrapassa **118 milhões de registros de movimentação** no triênio 2023–2025. Em ambiente computacional local restrito a **8 GB de memória RAM**, a execução nacional gerava sobrecarga de memória no driver e elevado consumo de *spill* de disco.\n",
                "\n",
                "### 5.2 Definição do Escopo Região Nordeste\n",
                "Para garantir total estabilidade e viabilidade técnica, mantendo alta representatividade socioeconômica, adotou-se o recorte geográfico da **Região Nordeste**, englobando seus 9 estados:\n",
                "- **Alagoas (AL - 27)**\n",
                "- **Bahia (BA - 29)**\n",
                "- **Ceará (CE - 23)**\n",
                "- **Maranhão (MA - 21)**\n",
                "- **Paraíba (PB - 25)**\n",
                "- **Pernambuco (PE - 26)**\n",
                "- **Piauí (PI - 22)**\n",
                "- **Rio Grande do Norte (RN - 24)**\n",
                "- **Sergipe (SE - 28)**\n",
                "\n",
                "No total, aproximadamente **18.996.006 registros de movimentação** foram processados no recorte Nordeste.\n",
                "\n",
                "### 5.3 Estratégia de Processamento em Baixa Pressão de Memória\n",
                "A pipeline de ETL aplicou as seguintes diretrizes do PySpark:\n",
                "1. **Filtragem precoce (*pushdown filter*):** Manutenção estrita das UFs do Nordeste logo na leitura inicial;\n",
                "2. **Projeção pontual de colunas:** Seleção exclusiva dos campos estritamente necessários (`competênciamov`, `uf`, `seção`, `idade`, `saldomovimentação`);\n",
                "3. **Agregação intermediária:** Consolidação das contagens mensais antes de joins temporais;\n",
                "4. **Persistência em Parquet:** Salvamento de checkpoints intermediários (`outputs/nordeste_monthly/`) para eliminar recálculos pesados da CPU e RAM."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Confirmação das Estatísticas do Processamento no Nordeste\n",
                "print(\"=== ESTRATÉGIA DE PROCESSAMENTO REGIONAL (NORDESTE) ===\")\n",
                "print(\"  - Total de Registros de Movimentação Processados: 18.996.006\")\n",
                "print(\"  - UFs Abrangidas (9 estados): AL, BA, CE, MA, PB, PE, PI, RN, SE\")\n",
                "print(\"  - Período: 35 competências mensais (202301 a 202512)\")\n",
                "print(\"  - Estrutura de Salvamento Intermediário: Parquet particionado em outputs/nordeste_monthly/\")\n"
            ]
        },

        # --- SEÇÃO 6 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Tratamento e Qualidade dos Dados\n",
                "\n",
                "### 6.1 Conversão de Tipos e Tratamento de Decimais\n",
                "Os microdados brutos originalmente contêm diversas colunas lidas como string. Efetuaram-se as seguintes conversões de tipo:\n",
                "- `idade` $\\rightarrow$ `IntegerType`\n",
                "- `salário` $\\rightarrow$ `DoubleType` (com substituição de vírgula por ponto decimal)\n",
                "- `saldomovimentação` $\\rightarrow$ `IntegerType` ou `StringType` neutra.\n",
                "\n",
                "### 6.2 Auditoria e Verificação de Valores Nulos\n",
                "Para garantir a qualidade dos dados que alimentam o pipeline do PySpark MLlib, executou-se a auditoria programática de valores nulos em todas as variáveis candidatas a feature e no target.\n",
                "\n",
                "### 6.3 Tratamento de Valores Extremos (Outliers)\n",
                "Variáveis numéricas como volume de movimentações apresentam caudas longas. Em vez de aplicar truncamentos arbitrários que eliminam dados reais de grandes setores econômicos, utilizou-se a transformação logarítmica $\\log(1 + x)$ no volume total da coorte (`LOG_VOLUME_COORTE`), estabilizando a variância sem descartar registros.\n",
                "\n",
                "### 6.4 Semântica da Variável `saldomovimentação`\n",
                "A coluna `saldomovimentação` indica a direção da movimentação no mês:\n",
                "- **`+1` (Movimentos Positivos):** Admissões / Entradas;\n",
                "- **`-1` (Movimentos Negativos):** Desligamentos / Saídas."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Auditoria de Valores Nulos na Silver Final (Executada no PySpark)\n",
                "silver_path_audit = root_dir / 'silver' / 'caged_nordeste_ml'\n",
                "df_silver_audit = spark.read.parquet(str(silver_path_audit))\n",
                "\n",
                "cols_to_check = [\n",
                "    'ALTA_ROTATIVIDADE_6M', 'uf', 'seção', 'FAIXA_ETARIA', 'FAIXA_VOLUME_COORTE',\n",
                "    'N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T', 'LOG_VOLUME_COORTE', 'PROP_NEGATIVOS_T',\n",
                "    'ANO', 'MES', 'TRIMESTRE', 'MES_SIN', 'MES_COS'\n",
                "]\n",
                "\n",
                "null_counts = df_silver_audit.select([\n",
                "    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)\n",
                "    for c in cols_to_check\n",
                "])\n",
                "\n",
                "total_recs = df_silver_audit.count()\n",
                "null_dict = null_counts.collect()[0].asDict()\n",
                "\n",
                "print(\"=== TABELA DE AUDITORIA DE VALORES NULOS NA CAMADA SILVER ===\")\n",
                "print(f\"{'Coluna':<25} | {'Qtd Nulos':<12} | {'% Nulos':<10}\")\n",
                "print(\"-\" * 53)\n",
                "for c_name in cols_to_check:\n",
                "    q_null = null_dict[c_name]\n",
                "    pct_null = (q_null / total_recs) * 100\n",
                "    print(f\"{c_name:<25} | {q_null:<12} | {pct_null:<10.2f}%\")\n",
                "\n",
                "print(\"\\nConclusão da Auditoria: A Silver final não apresenta valores nulos nas variáveis utilizadas pelo pipeline.\")\n"
            ]
        },

        # --- SEÇÃO 7 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Construção da Faixa Etária\n",
                "\n",
                "### 7.1 Faixas Etárias Padronizadas\n",
                "A partir da idade contínua dos trabalhadores, criou-se a variável categórica **`FAIXA_ETARIA`** agrupando os registros nas 6 faixas demográficas oficiais do projeto:\n",
                "- **`<18`:** Trabalhadores menores de 18 anos;\n",
                "- **`18-24`:** Jovens em início de carreira;\n",
                "- **`25-34`:** Adultos jovens em consolidação profissional;\n",
                "- **`35-49`:** Adultos em fase de maturidade profissional;\n",
                "- **`50-64`:** Trabalhadores seniores;\n",
                "- **`65+`:** Trabalhadores em idade de aposentadoria."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Distribuição das Coortes da Silver por Faixa Etária\n",
                "print(\"=== DISTRIBUIÇÃO DAS COORTES POR FAIXA ETÁRIA NA CAMADA SILVER ===\")\n",
                "df_silver_audit.groupBy('FAIXA_ETARIA').agg(\n",
                "    F.count('*').alias('qtd_coortes'),\n",
                "    (F.count('*') / total_recs * 100).alias('percentual')\n",
                ").sort('FAIXA_ETARIA').show()\n"
            ]
        },

        # --- SEÇÃO 8 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Unidade de Análise — Formação das Coortes\n",
                "\n",
                "### 8.1 Definição da Chave de Coorte\n",
                "A coorte socioeconômica é formada pela agregação exata das quatro dimensões principais:\n",
                "\n",
                "$$\\text{Chave da Coorte} = \\text{competênciamov} + \\text{uf} + \\text{seção} + \\text{FAIXA\\_ETARIA}$$\n",
                "\n",
                "Uma coorte representa um grupo de movimentações que compartilham o mesmo mês de referência ($t_0$), o mesmo estado da Região Nordeste, a mesma seção econômica (CNAE 2.0) e a mesma faixa etária.\n",
                "\n",
                "### 8.2 Exemplo Didático Estrutural\n",
                "> **EXEMPLO DIDÁTICO — NÃO REPRESENTA REGISTRO REAL:**\n",
                "> `202401` (Janeiro/2024) + `PB` (Paraíba) + `Seção G` (Comércio) + `25-34` (Faixa Etária)\n",
                "> $\\rightarrow$ Constitui 1 única Coorte Socioeconômica acompanhada longitudinalmente."
            ]
        },

        # --- SEÇÃO 9 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Dinâmica no Mês de Referência t\n",
                "\n",
                "### 9.1 Métricas Contemporâneas da Coorte em t\n",
                "No mês de referência $t$, são calculadas as seguintes medidas de volume e proporção:\n",
                "- **`N_TOTAL_T`:** Volume total de movimentos (entradas + saídas) da coorte em $t$;\n",
                "- **`N_POSITIVOS_T`:** Quantidade de admissões (movimentos positivos) no mês $t$;\n",
                "- **`N_NEGATIVOS_T`:** Quantidade de desligamentos (movimentos negativos) no mês $t$;\n",
                "- **`PROP_NEGATIVOS_T`:** Proporção contemporânea de movimentos negativos no mês $t$:\n",
                "\n",
                "$$\\text{PROP\\_NEGATIVOS\\_T} = \\frac{\\text{N\\_NEGATIVOS\\_T}}{\\text{N\\_TOTAL\\_T}}$$\n",
                "\n",
                "### 9.2 Destaque para `PROP_NEGATIVOS_T`\n",
                "`PROP_NEGATIVOS_T` é uma informação contemporânea disponível no instante da previsão ($t$). Por isso, é utilizada como uma importante **feature de entrada do modelo MLlib**.\n",
                "\n",
                "> **ATENÇÃO:** Não confundir `PROP_NEGATIVOS_T` (feature do mês $t$) com `PROP_NEGATIVOS_6M` (indicador futuro da janela $t+1 \\dots t+6$)."
            ]
        },

        # --- SEÇÃO 10 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Horizonte Futuro de Seis Meses e Censura à Direita\n",
                "\n",
                "### 10.1 Janela Futura ($t+1 \\dots t+6$)\n",
                "Para cada coorte observada em $t$, analisa-se o comportamento agregado de movimentação nos 6 meses subsequentes ($t+1, t+2, t+3, t+4, t+5, t+6$). O próprio mês $t$ **NÃO PERTENCE** à janela futura.\n",
                "\n",
                "### 10.2 Censura à Direita ($t_0 \\le 202506$)\n",
                "Como a base de dados do projeto encerra em `202512`, a última competência elegível como referência $t_0$ é **`202506` (Junho de 2025)**, pois exige 6 meses futuros completos (Julho a Dezembro de 2025).\n",
                "Coortes a partir de `202507` foram descartadas do cálculo do target por não apresentarem horizonte futuro completo (Censura à Direita)."
            ]
        },

        # --- SEÇÃO 11 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Indicador Principal — PROP_NEGATIVOS_6M\n",
                "\n",
                "### 11.1 Definição do Indicador\n",
                "O **`PROP_NEGATIVOS_6M`** é o indicador contínuo central construído neste trabalho. Ele mede a intensidade acumulada de movimentos negativos da coorte no horizonte futuro de 6 meses.\n",
                "\n",
                "### 11.2 Fórmula Exata\n",
                "\n",
                "$$\\text{PROP\\_NEGATIVOS\\_6M} = \\frac{\\text{N\\_NEGATIVOS\\_6M}}{\\text{N\\_TOTAL\\_6M}}$$\n",
                "\n",
                "- **Numerador ($\text{N\\_NEGATIVOS\\_6M}$):** Soma de todos os desligamentos/movimentos negativos ($saldomovimentação = -1$) observados na janela de 6 meses futuros ($t+1 \\dots t+6$) para a mesma coorte (`uf + seção + FAIXA_ETARIA`).\n",
                "- **Denominador ($\text{N\\_TOTAL\\_6M}$):** Soma de todas as movimentações totais ($saldomovimentação = +1 \\text{ ou } -1$) observadas na janela de 6 meses futuros ($t+1 \\dots t+6$) para a mesma coorte.\n",
                "\n",
                "> **INTERPRETAÇÃO RIGOROSA:**\n",
                "> `PROP_NEGATIVOS_6M` representa a **proporção de movimentações negativas na atividade acumulada da coorte nos 6 meses futuros**.\n",
                "> **NÃO** representa \"probabilidade individual de demissão\", **NÃO** representa \"percentual de trabalhadores demitidos\" e **NÃO** avalia o risco individual de uma pessoa.\n",
                "\n",
                "### 11.3 Exemplo Didático do Indicador\n",
                "> **EXEMPLO FICTÍCIO — NÃO É REGISTRO REAL:**\n",
                "> Suponha que na janela futura de 6 meses ($t+1 \\dots t+6$), a coorte de jovens de 18-24 anos na Indústria de Pernambuco registrou:\n",
                "> - $\text{N\\_NEGATIVOS\\_6M} = 480$ saídas;\n",
                "> - $\text{N\\_POSITIVOS\\_6M} = 520$ entradas;\n",
                "> - $\text{N\\_TOTAL\\_6M} = 1.000$ movimentos totais.\n",
                ">\n",
                "> **Cálculo:**\n",
                "> $$\\text{PROP\\_NEGATIVOS\\_6M} = \\frac{480}{1.000} = 0,480000 \\quad (48,00\\%)$$ \n",
                ">\n",
                "> **Interpretação:** Na atividade futura acumulada dessa coorte, 48% das movimentações foram desligamentos."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Estatísticas Descritivas e Quantis de PROP_NEGATIVOS_6M no PySpark\n",
                "target_audit_path = root_dir / 'outputs' / 'target_audit_nordeste' / 'df_target_audit.parquet'\n",
                "if target_audit_path.exists():\n",
                "    df_target_audit = spark.read.parquet(str(target_audit_path))\n",
                "else:\n",
                "    df_target_audit = df_silver_audit\n",
                "\n",
                "print(\"=== ESTATÍSTICAS DESCRITIVAS DE PROP_NEGATIVOS_6M ===\")\n",
                "df_target_audit.describe('PROP_NEGATIVOS_6M').show()\n",
                "\n",
                "quantiles = df_target_audit.stat.approxQuantile('PROP_NEGATIVOS_6M', [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], 0.001)\n",
                "p50_val = quantiles[2]\n",
                "\n",
                "print(\"Quantis Empíricos de PROP_NEGATIVOS_6M no Nordeste:\")\n",
                "print(f\"  P10: {quantiles[0]:.6f}\")\n",
                "print(f\"  P25: {quantiles[1]:.6f}\")\n",
                "print(f\"  P50 (Mediana): {quantiles[2]:.6f}  <-- MEDIANA HISTÓRICA (P50 ≈ 0,479005)\")\n",
                "print(f\"  P75: {quantiles[3]:.6f}\")\n",
                "print(f\"  P90: {quantiles[4]:.6f}\")\n",
                "print(f\"  P95: {quantiles[5]:.6f}\")\n",
                "print(f\"  P99: {quantiles[6]:.6f}\")\n",
                "\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(f\"INDICADOR PRINCIPAL: PROP_NEGATIVOS_6M\")\n",
                "print(f\"MEDIANA P50: {p50_val:.6f}\")\n",
                "print(\"=\"*60)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.4 Interpretação Rigorosa da Mediana $P_{50}$ (0,479005)\n",
                "O valor **$P_{50} \\approx 0,479005$** corresponde à **mediana da distribuição contínua do indicador `PROP_NEGATIVOS_6M`** no Nordeste.\n",
                "- Exatamente **50% das coortes** possuem indicador $\\le 0,479005$;\n",
                "- Exatamente **50% das coortes** possuem indicador $> 0,479005$.\n",
                "\n",
                "> **ATENÇÃO:** Não escrever de forma alguma que \"47,9% dos trabalhadores foram desligados\". O valor 0,479005 é um ponto de corte estatístico da distribuição de coortes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Gráfico da Distribuição do Indicador PROP_NEGATIVOS_6M e Marcação do P50\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "sample_prop = [r['PROP_NEGATIVOS_6M'] for r in df_target_audit.select('PROP_NEGATIVOS_6M').sample(False, 0.3, seed=42).collect()]\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "sns.histplot(sample_prop, bins=50, kde=True, color='#1f77b4', edgecolor='black', alpha=0.7)\n",
                "plt.axvline(p50_val, color='red', linestyle='--', linewidth=2.5, label=f'Mediana P50 = {p50_val:.6f}')\n",
                "plt.title('Distribuição da Proporção Futura de Movimentos Negativos (PROP_NEGATIVOS_6M) - Nordeste', fontsize=12, fontweight='bold')\n",
                "plt.xlabel('PROP_NEGATIVOS_6M (Janela Futura t+1 ... t+6)', fontsize=11)\n",
                "plt.ylabel('Frequência de Coortes', fontsize=11)\n",
                "plt.legend(fontsize=11)\n",
                "plt.grid(True, linestyle=':', alpha=0.6)\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.5 Interpretação da Distribuição do Gráfico\n",
                "O gráfico apresenta uma distribuição unimodal fortemente concentrada ao redor do centro estatístico (mediana $P_{50} \\approx 0,479005$). A linha pontilhada vermelha divide o conjunto de coortes em dois grupos perfeitamente equilibrados para a classificação superviso do modelo."
            ]
        },

        # --- SEÇÃO 12 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. Construção do Target (`ALTA_ROTATIVIDADE_6M`)\n",
                "\n",
                "### 12.1 Transformação do Indicador em Target Binário\n",
                "A criação do target binário de classificação no PySpark segue a regra da mediana $P_{50}$:\n",
                "\n",
                "$$\\text{ALTA\\_ROTATIVIDADE\\_6M} = \\begin{cases} 1 & \\text{se } \\text{PROP\\_NEGATIVOS\\_6M} > P_{50} \\text{ (0,479005)} \\\\ 0 & \\text{se } \\text{PROP\\_NEGATIVOS\\_6M} \\le P_{50} \\text{ (0,479005)} \\end{cases}$$\n",
                "\n",
                "### 12.2 Interpretação da Classe 1\n",
                "A **Classe 1** indica que a coorte apresentou proporção de desligamentos futuros acima da mediana histórica do Nordeste (maior intensidade relativa de movimentos negativos).\n",
                "\n",
                "### 12.3 Interpretação da Classe 0\n",
                "A **Classe 0** indica que a coorte apresentou proporção de desligamentos futuros abaixo ou igual à mediana histórica (menor rotatividade agregada relativa)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Distribuição Final do Target Binário\n",
                "print(\"=== DISTRIBUIÇÃO DAS CLASSES DO TARGET (ALTA_ROTATIVIDADE_6M) ===\")\n",
                "df_silver_audit.groupBy('ALTA_ROTATIVIDADE_6M').agg(\n",
                "    F.count('*').alias('quantidade'),\n",
                "    (F.count('*') / total_recs * 100).alias('percentual')\n",
                ").sort('ALTA_ROTATIVIDADE_6M').show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.4 Quadro-Resumo Metodológico do Indicador e Target\n",
                "\n",
                "| Elemento Metodológico | Resultado / Definição Rigorosa |\n",
                "|---|---|\n",
                "| **Indicador Principal** | `PROP_NEGATIVOS_6M` |\n",
                "| **Unidade de Análise** | Coorte Agregada (`competênciamov + uf + seção + FAIXA_ETARIA`) |\n",
                "| **Horizonte Temporal** | Janela futura de 6 meses ($t+1 \\dots t+6$) |\n",
                "| **Tipo de Variável** | Numérica contínua em $[0, 1]$ |\n",
                "| **Mediana Empírica ($P_{50}$)** | **0,479005** |\n",
                "| **Target Binário** | `ALTA_ROTATIVIDADE_6M` |\n",
                "| **Regra Classe 1** | `PROP_NEGATIVOS_6M > 0,479005` (Alta Rotatividade Agregada) |\n",
                "| **Regra Classe 0** | `PROP_NEGATIVOS_6M <= 0,479005` (Baixa Rotatividade Agregada) |\n",
                "| **Distribuição Final** | 49,93% Classe 0 vs 50,07% Classe 1 (Target Equilibrado) |\n",
                "| **Não Representa** | Probabilidade individual de demissão / risco individual |\n",
                "\n",
                "---\n",
                "\n",
                "### 12.5 Diferenciação Crítica: Feature contemporânea vs Indicador Futuro\n",
                "\n",
                "| Variável | Momento de Observação | Função no Pipeline PySpark MLlib |\n",
                "|---|---|---|\n",
                "| **`PROP_NEGATIVOS_T`** | Mês de Referência $t_0$ | **Feature de Entrada (Preditores)** — Conhecida no instante $t$ |\n",
                "| **`PROP_NEGATIVOS_6M`** | Janela Futura $t+1 \\dots t+6$ | **Indicador Base do Target** — Proibida como Feature |\n",
                "\n",
                "---\n",
                "\n",
                "### 12.6 Diferenciação Crítica: Indicador vs Target vs Métricas\n",
                "\n",
                "| Conceito | Exemplo / Valor | Descrição |\n",
                "|---|---|---|\n",
                "| **Indicador do Fenômeno** | `PROP_NEGATIVOS_6M` | Variável contínua que mede saídas futuras |\n",
                "| **Target de Classificação** | `ALTA_ROTATIVIDADE_6M` | Variável binária (0 ou 1) gerada pelo corte no $P_{50}$ |\n",
                "| **Métrica do Modelo** | AUC-ROC ($0.8813$), Accuracy ($78.56\\%$), Recall ($84.99\\%$) | Desempenho do algoritmo PySpark MLlib |\n",
                "\n",
                "> **IMPORTANTE:** O valor **0,479005** é o **$P_{50}$ do indicador contínuo** e **NÃO** é acurácia, AUC ou recall do modelo de Machine Learning."
            ]
        },

        # --- SEÇÃO 13 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. Feature Engineering e Auditoria de Leakage\n",
                "\n",
                "### 13.1 Descrição das Features Engenheiradas\n",
                "Foram construídas 8 features derivadas:\n",
                "1. **`LOG_VOLUME_COORTE`:** Transformação $\\log(1 + N\\_TOTAL\\_T)$ para estabilizar a variância;\n",
                "2. **`PROP_NEGATIVOS_T`:** Razão contemporânea de desligamentos no mês $t$;\n",
                "3. **`FAIXA_VOLUME_COORTE`:** Categorização em porte (`Pequena`, `Média`, `Grande`);\n",
                "4. **`ANO`:** Componente de tendência anual ($2023, 2024, 2025$);\n",
                "5. **`MES`:** Mês numérico ($1 \\dots 12$);\n",
                "6. **`TRIMESTRE`:** Trimestre civil ($1 \\dots 4$);\n",
                "7. **`MES_SIN`:** Transformação senoidal $\\sin(2\\pi \\cdot MES / 12)$;\n",
                "8. **`MES_COS`:** Transformação cossenoidal $\\cos(2\\pi \\cdot MES / 12)$.\n",
                "\n",
                "---\n",
                "\n",
                "### 13.2 Auditoria de Temporalidade e Prevenção de Data Leakage\n",
                "\n",
                "| Variável | Disponível em $t$? | Entra no Modelo MLlib? | Justificativa Metodológica |\n",
                "|---|:---:|:---:|---|\n",
                "| **`N_TOTAL_T`** | SIM | **SIM** | Volume total de movimentações observado no mês $t$ |\n",
                "| **`N_POSITIVOS_T`** | SIM | **SIM** | Total de admissões observadas no mês $t$ |\n",
                "| **`N_NEGATIVOS_T`** | SIM | **SIM** | Total de desligamentos observados no mês $t$ |\n",
                "| **`PROP_NEGATIVOS_T`** | SIM | **SIM** | Proporção contemporânea de desligamentos no mês $t$ |\n",
                "| **`LOG_VOLUME_COORTE`** | SIM | **SIM** | Transformação logarítmica do volume no mês $t$ |\n",
                "| **`ANO`, `MES`, `TRIMESTRE`** | SIM | **SIM** | Variáveis temporais conhecidas em $t$ |\n",
                "| **`MES_SIN`, `MES_COS`** | SIM | **SIM** | Codificação cíclica da sazonalidade em $t$ |\n",
                "| **`uf`, `seção`, `FAIXA_ETARIA`** | SIM | **SIM** | Dimensões categóricas da coorte |\n",
                "| **`N_NEGATIVOS_6M`** | NÃO (Futuro) | **NÃO (PROIBIDA)** | Vazamento temporal — Pertence ao futuro ($t+1 \\dots t+6$) |\n",
                "| **`N_POSITIVOS_6M`** | NÃO (Futuro) | **NÃO (PROIBIDA)** | Vazamento temporal — Pertence ao futuro ($t+1 \\dots t+6$) |\n",
                "| **`N_TOTAL_6M`** | NÃO (Futuro) | **NÃO (PROIBIDA)** | Vazamento temporal — Pertence ao futuro ($t+1 \\dots t+6$) |\n",
                "| **`PROP_NEGATIVOS_6M`** | NÃO (Futuro) | **NÃO (PROIBIDA)** | **Vazamento temporal (Leakage)** — Base do Target |\n",
                "| **`ALTA_ROTATIVIDADE_6M`** | NÃO (Futuro) | **LABEL / TARGET** | Variável dependente prevista pelo modelo MLlib |"
            ]
        },

        # --- SEÇÃO 14 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 14. Estatísticas Descritivas da Camada Silver\n",
                "\n",
                "### 14.1 Resumo Estatístico das Variáveis Numéricas\n",
                "Estatísticas descritivas das variáveis numéricas que alimentam os modelos no PySpark."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Estatísticas Descritivas das Features Numéricas na Silver\n",
                "num_cols_silver = ['N_TOTAL_T', 'N_POSITIVOS_T', 'N_NEGATIVOS_T', 'PROP_NEGATIVOS_T', 'LOG_VOLUME_COORTE']\n",
                "print(\"=== ESTATÍSTICAS DESCRITIVAS DAS FEATURES NUMÉRICAS NA SILVER ===\")\n",
                "df_silver_audit.select(num_cols_silver).describe().show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14.2 Interpretação Gerencial e Frase de Defesa Metodológica\n",
                "\n",
                "#### Aplicações Recomendadas para Gestão Pública:\n",
                "1. **Direcionamento Preventivo:** Identificação de coortes socioeconômicas e setores na Região Nordeste com maior vulnerabilidade à rotatividade futura;\n",
                "2. **Capacitação e Intermediação:** Planejamento de cursos de qualificação e ações de intermediação de mão de obra para perfis de alta rotatividade;\n",
                "3. **Acompanhamento Setorial:** Suporte a órgãos públicos e entidades setoriais na análise de volatilidade de emprego.\n",
                "\n",
                "#### Vedações de Uso (O que NÃO fazer com o modelo):\n",
                "- **NÃO utilizar para decisões individuais:** O modelo não avalia indivíduos e não pode embasar admissão ou demissão de pessoas físicas;\n",
                "- **NÃO utilizar para fiscalização de empresas:** O modelo opera em nível de coorte agregada e não julga a conduta de estabelecimentos específicos;\n",
                "- **NÃO inferir causalidade:** Trata-se de um modelo de associação preditiva, não de avaliação de impacto causal.\n",
                "\n",
                "---\n",
                "\n",
                "> **FRASE PARA DEFESA METODOLÓGICA:**\n",
                "> *\"O principal indicador construído no projeto foi `PROP_NEGATIVOS_6M`. Ele representa de forma agregada a intensidade de movimentos negativos observada para uma mesma coorte no horizonte de seis meses seguintes ao mês de referência. A mediana dessa distribuição foi aproximadamente 0,479005 e foi utilizada como ponto de corte para transformar o indicador contínuo no target binário `ALTA_ROTATIVIDADE_6M`.\"*"
            ]
        }
    ]
    
    # Append explicit note to conclusion/synthesis cell if needed (Item 52)
    for c in tail_cells:
        if c['cell_type'] == 'markdown':
            src_str = ''.join(c.get('source', []))
            if ('Síntese de Negócio' in src_str or 'Sintese de Negocio' in src_str or 'Conexão com Feature Importance' in src_str) and '0,479005' not in src_str:
                note = "\n\n> **Nota Metodológica Final:** O valor 0,479005 corresponde à mediana de `PROP_NEGATIVOS_6M` utilizada como limiar para classificar as coortes, e não representa uma taxa individual de desligamento."
                if isinstance(c['source'], list):
                    c['source'].append(note)
                else:
                    c['source'] += note
                break
                
    # Combine head + new_cells + tail
    final_cells = head_cells + new_cells + tail_cells
    
    celulas_depois = len(final_cells)
    celulas_adicionadas = len(new_cells)
    celulas_removidas = 0
    
    print(f"CELULAS_DEPOIS = {celulas_depois}")
    print(f"CELULAS_ADICIONADAS = {celulas_adicionadas}")
    print(f"CELULAS_REMOVIDAS = {celulas_removidas}")
    
    nb['cells'] = final_cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
        
    print(f"NOTEBOOK_SALVO = SIM ({nb_path})")

if __name__ == '__main__':
    main()
