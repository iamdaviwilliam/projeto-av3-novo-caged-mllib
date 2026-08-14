# encoding: utf-8
import json
import os
import sys
import io
import contextlib
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
hadoop_home = (root_dir / 'hadoop').resolve()
os.environ['HADOOP_HOME'] = str(hadoop_home)
os.environ['PATH'] = str(hadoop_home / 'bin') + os.pathsep + os.environ.get('PATH', '')

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    nb_path = root_dir / 'notebooks' / 'pipeline_mllib.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    spark = SparkSession.builder \
        .appName('ExecuteNewCells') \
        .master('local[2]') \
        .config('spark.driver.memory', '2g') \
        .config('spark.sql.adaptive.enabled', 'true') \
        .getOrCreate()

    new_code_indices = [5, 8, 10, 12, 17, 22, 26]

    exec_globals = {'spark': spark, 'F': F, 'root_dir': root_dir}

    for idx in new_code_indices:
        cell = nb['cells'][idx]
        if cell['cell_type'] != 'code':
            continue
        code = ''.join(cell['source'])
        print(f"Executing new code cell {idx}...")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, exec_globals)
            out_text = buf.getvalue()
            cell['outputs'] = [{
                'name': 'stdout',
                'output_type': 'stream',
                'text': out_text.splitlines(keepends=True)
            }]
            print(f"Cell {idx} executed successfully ({len(out_text)} bytes output).")
        except Exception as e:
            print(f"Error executing cell {idx}: {e}")

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    print("New code cells executed and saved.")

if __name__ == '__main__':
    main()
