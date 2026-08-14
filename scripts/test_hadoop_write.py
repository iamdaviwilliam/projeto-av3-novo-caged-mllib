# encoding: utf-8
"""
Script de validação da correção do HADOOP_HOME e winutils.exe no Windows.
"""
import os
import sys
import shutil
from pathlib import Path

# Configuração de HADOOP_HOME e PATH ANTES de importar/iniciar o PySpark
root_dir = Path(__file__).resolve().parent.parent
hadoop_home = (root_dir / "hadoop").resolve()
os.environ["HADOOP_HOME"] = str(hadoop_home)
os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

print(f"HADOOP_HOME configurado em: {os.environ['HADOOP_HOME']}")
print(f"winutils.exe existe: {(hadoop_home / 'bin' / 'winutils.exe').exists()}")

from pyspark.sql import SparkSession

def main():
    print("\n=== 1. Inicializando SparkSession com HADOOP_HOME ativo ===")
    spark = SparkSession.builder \
        .appName("TestHadoopWinutils") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    print(f"Spark Version: {spark.version}")
    try:
        hadoop_ver = spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion()
        print(f"Hadoop Version no Spark: {hadoop_ver}")
    except Exception as e:
        print(f"Hadoop Version: {e}")

    print("\n=== 2. Teste Mínimo de Escrita em Parquet (spark.range(10)) ===")
    test_out = root_dir / "outputs" / "test_parquet"
    
    print(f"Escrita em: {test_out}")
    spark.range(10).write.mode("overwrite").parquet(str(test_out))
    print("SUCESSO: Escrita em Parquet concluída com mode('overwrite')!")

    print("\n=== 3. Leitura de Validação do Parquet ===")
    df_read = spark.read.parquet(str(test_out))
    df_read.show()
    print(f"Total de registros lidos no teste mínimo: {df_read.count()}")

    spark.stop()
    print("SparkSession encerrada.")

if __name__ == "__main__":
    main()
