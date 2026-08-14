# Diretório de Dados — Novo CAGED (2023–2025)

## 1. Fonte dos Microdados
Os microdados utilizados pertencem ao **Novo CAGED (Cadastro Geral de Empregados e Desempregados)**, disponibilizados mensalmente pelo **Ministério do Trabalho e Emprego (MTE)** através do FTP público do governo federal e do Portal de Dados Abertos.

- **Servidor FTP Oficial:** `ftp://ftp.mtps.gov.br/pdet/microdados/CAGED/`
- **Tabelas Utilizadas:** Declarações de Movimentação Mensal (`CAGEDMOV2023*.7z`, `CAGEDMOV2024*.7z`, `CAGEDMOV2025*.7z`).

## 2. Política de Versionamento no Git
Em atendimento às diretrizes do repositório e limites do GitHub, **os arquivos brutos compactados (`.7z`) e descompactados (`.txt`) NÃO são versionados no Git**, estando devidamente incluídos no `.gitignore`.

Os dados intermediários e a camada **Silver final (`silver/caged_nordeste_ml/`)** de tamanho reduzido (0,75 MB) encontram-se persistidos e versionados no repositório para garantir reprodutibilidade imediata da modelagem.

## 3. Período e Arquivos Efetivamente Utilizados
- **Triênio Observado:** 2023 a 2025 (35 arquivos mensais de movimentação descompactados);
- **Competências Elegíveis para Modelagem:** `202301` a `202506` (respeitando a censura à direita de 6 meses futuros até `202512`);
- **Escopo Geográfico:** **Região Nordeste** (Estados: AL, BA, CE, MA, PB, PE, PI, RN, SE; Códigos IBGE: 27, 29, 23, 21, 25, 26, 22, 24, 28).

## 4. Instruções para Download e Reconstrução Local (Opcional)
Caso o usuário deseje reexecutar a ingestão dos dados brutos a partir da estaca zero:
1. Baixe os arquivos `.7z` das movimentações de 2023 a 2025 do FTP do MTE;
2. Extraia os arquivos `.txt` no diretório `data/raw/` (ou na raiz `data/`);
3. Execute o script de filtragem precoce `scripts/generate_and_execute_nordeste_notebook.py`.
