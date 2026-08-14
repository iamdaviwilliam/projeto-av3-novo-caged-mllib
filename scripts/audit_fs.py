import os
import json
import shutil
from pathlib import Path
from pyspark.sql import SparkSession

root = Path('.').resolve()
print(f"1. ROOT_PROJETO = {root}")

# Setup Hadoop Home
hadoop_home = root / "hadoop"
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

# 1. Inspect Directories
print("\n=== 1. ESTRUTURA ATUAL DAS PASTAS DO PROJETO ===")
dirs_to_check = ["data", "notebooks", "outputs", "silver", "models", "scripts"]
for d in dirs_to_check:
    dp = root / d
    if dp.exists():
        files = [f for f in dp.rglob('*') if f.is_file()]
        print(f"Diretório '{d}/': EXISTE ({len(files)} arquivos no total)")
    else:
        print(f"Diretório '{d}/': NÃO EXISTE")

# 2. Search for Parquets
print("\n=== 2. PARQUETS ENCONTRADOS FISICAMENTE EM TODO O PROJETO ===")
found_parquets = []
for p in root.rglob("*"):
    if p.is_dir() and (p.name.endswith(".parquet") or list(p.glob("part-*.parquet"))):
        rel = p.relative_to(root)
        part_files = list(p.glob("part-*.parquet"))
        has_success = (p / "_SUCCESS").exists()
        total_size = sum(f.stat().st_size for f in p.iterdir() if f.is_file())
        found_parquets.append({
            "path_rel": str(rel),
            "path_abs": str(p),
            "has_success": has_success,
            "num_parts": len(part_files),
            "size_bytes": total_size
        })
        print(f"Parquet: {rel}")
        print(f"  Caminho Absoluto: {p}")
        print(f"  Tamanho: {total_size:,} bytes")
        print(f"  Arquivos part-*: {len(part_files)}")
        print(f"  Possui _SUCCESS: {has_success}")

if not found_parquets:
    print("Nenhum diretório Parquet encontrado.")

# 3. Check Notebook Paths
print("\n=== 3. CAMINHOS UTILIZADOS NO NOTEBOOK ===")
nb_path = root / "notebooks" / "pipeline_mllib.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if ".write" in source or ".parquet" in source:
            print(f"\n--- Célula #{idx+1} ---")
            for line in cell["source"]:
                if any(k in line for k in [".write", ".parquet", "Path(", "out_"]):
                    print("  ", line.strip())

# 4. Check Resolution of Relative Paths
print("\n=== 4. RESOLUÇÃO DE DIRETÓRIO E CAMINHOS RELATIVOS ===")
print(f"Notebook File Location: {nb_path}")
print(f"Python Kernel Root CWD: {root}")
p_out_rel = Path("outputs/target_audit/df_target_audit.parquet")
p_out_rel_dotdot = Path("../outputs/target_audit/df_target_audit.parquet")
print(f"Exemplo 'outputs/target_audit/...': { (root / p_out_rel).resolve() }")
print(f"Exemplo se executado de dentro de notebooks/ ('../outputs/...'): { (nb_path.parent / p_out_rel_dotdot).resolve() }")

# 5. Check Target Parquets
print("\n=== 5. VERIFICAÇÃO ESPECÍFICA DOS PARQUETS DO TARGET ===")
p_audit = root / "outputs" / "target_audit" / "df_target_audit.parquet"
p_model = root / "outputs" / "target_audit" / "df_modelagem.parquet"

for name, path in [("df_target_audit.parquet", p_audit), ("df_modelagem.parquet", p_model)]:
    print(f"\nTarget Parquet: {name}")
    if path.exists():
        parts = list(path.glob("part-*.parquet"))
        has_succ = (path / "_SUCCESS").exists()
        tot_sz = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
        print(f"  EXISTE: SIM")
        print(f"  CAMINHO ABSOLUTO: {path}")
        print(f"  TAMANHO: {tot_sz:,} bytes")
        print(f"  QUANTIDADE DE ARQUIVOS PART: {len(parts)}")
        print(f"  POSSUI _SUCCESS: {'SIM' if has_succ else 'NÃO'}")
    else:
        print(f"  EXISTE: NÃO")
        print(f"  CAMINHO ABSOLUTO: {path}")

# 6. Check Silver
print("\n=== 6. VERIFICAÇÃO DA CAMADA SILVER ===")
p_silver = root / "silver" / "df_silver.parquet"
print(f"Silver Folder: {root / 'silver'}")
if p_silver.exists():
    parts = list(p_silver.rglob("part-*.parquet"))
    has_succ = (p_silver / "_SUCCESS").exists()
    tot_sz = sum(f.stat().st_size for f in p_silver.rglob("*") if f.is_file())
    print(f"  SILVER PARQUET EXISTE: SIM")
    print(f"  CAMINHO ABSOLUTO: {p_silver}")
    print(f"  TAMANHO: {tot_sz:,} bytes")
    print(f"  QUANTIDADE DE ARQUIVOS PART: {len(parts)}")
    print(f"  POSSUI _SUCCESS: {'SIM' if has_succ else 'NÃO'}")
else:
    print(f"  SILVER PARQUET EXISTE: NÃO")

# 7. Test Min Write
print("\n=== 7. TESTE MÍNIMO DE ESCRITA NO SPARK ===")
spark = SparkSession.builder.appName("AuditFS").master("local[2]").config("spark.driver.memory", "2g").getOrCreate()
test_path = root / "outputs" / "test_parquet"
if test_path.exists():
    shutil.rmtree(test_path)

df_test = spark.range(10)
df_test.write.mode("overwrite").parquet(str(test_path))

test_exists = test_path.exists()
test_parts = list(test_path.glob("part-*.parquet")) if test_exists else []
test_succ = (test_path / "_SUCCESS").exists() if test_exists else False

print(f"outputs/test_parquet criado? {test_exists}")
print(f"Arquivos part-*: {len(test_parts)}")
print(f"_SUCCESS presente? {test_succ}")
read_cnt = spark.read.parquet(str(test_path)).count()
print(f"Leitura de confirmação (count): {read_cnt}")

spark.stop()
