# encoding: utf-8
"""
Script de diagnóstico do ambiente Hadoop/Spark no Windows.
"""
import os
import sys
import subprocess
import pyspark
from pyspark.sql import SparkSession
from pathlib import Path

def main():
    print("=== 1. DIAGNÓSTICO DO AMBIENTE ===")
    print(f"1. Versão do PySpark: {pyspark.__version__}")

    try:
        res = subprocess.run(["java", "-version"], capture_output=True, text=True)
        java_ver = res.stderr.splitlines()[0] if res.stderr else res.stdout
        print(f"2. Versão do Java: {java_ver}")
    except Exception as e:
        print(f"2. Versão do Java: Erro ({e})")

    spark = SparkSession.builder \
        .appName("DiagnosticoHadoop") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    print(f"3. spark.version: {spark.version}")

    try:
        hadoop_ver = spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion()
        print(f"4. Versão do Hadoop no Spark: {hadoop_ver}")
    except Exception as e:
        print(f"4. Versão do Hadoop no Spark: Erro ({e})")

    hh = os.environ.get("HADOOP_HOME")
    print(f"5. HADOOP_HOME: {hh}")

    path_env = os.environ.get("PATH", "")
    has_hadoop_in_path = "hadoop" in path_env.lower() or (hh is not None and hh in path_env)
    print(f"6. PATH contém HADOOP_HOME/bin: {has_hadoop_in_path}")

    winutils_found = False
    if hh:
        wpath = Path(hh) / "bin" / "winutils.exe"
        winutils_found = wpath.exists()
        print(f"7. winutils.exe em HADOOP_HOME/bin: {winutils_found} ({wpath})")
    else:
        print("7. winutils.exe: HADOOP_HOME não está configurado nas variáveis de ambiente.")

    try:
        default_fs = spark.sparkContext._jsc.hadoopConfiguration().get("fs.defaultFS")
        print(f"8. spark.hadoop.fs.defaultFS: {default_fs}")
    except Exception as e:
        print(f"8. spark.hadoop.fs.defaultFS: Erro ({e})")

    print("\n=== 9. TESTE MÍNIMO DE ESCRITA PARQUET (spark.range(10)) ===")
    test_dir = Path("../outputs/test_parquet")
    test_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        spark.range(10).write.mode("overwrite").parquet(str(test_dir))
        print("RESULTADO DO TESTE MÍNIMO: SUCESSO na escrita sem erro de checkHadoopHome!")
    except Exception as e:
        print(f"RESULTADO DO TESTE MÍNIMO: FALHOU com exceção:\n{e}")
        if "checkHadoopHome" in str(e) or "HADOOP_HOME" in str(e):
            print("-> CONFIRMADO: A falha é devido à falta do HADOOP_HOME / winutils.exe no ambiente Windows!")

    spark.stop()

if __name__ == "__main__":
    main()
