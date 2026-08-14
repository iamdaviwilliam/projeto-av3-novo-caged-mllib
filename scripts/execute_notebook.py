# encoding: utf-8
"""
Script oficial para executar e preservar os outputs do notebook notebooks/pipeline_mllib.ipynb.
"""

import json
from pathlib import Path
import io
import os
from contextlib import redirect_stdout, redirect_stderr

root_dir = Path(__file__).resolve().parent.parent
nb_path = root_dir / "notebooks" / "pipeline_mllib.ipynb"

def main():
    # Configuração de HADOOP_HOME para Windows antes da execução do notebook
    hadoop_home = (root_dir / "hadoop").resolve()
    os.environ["HADOOP_HOME"] = str(hadoop_home)
    os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

    print(f"Carregando notebook: {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    exec_globals = {}
    orig_cwd = os.getcwd()
    os.chdir(nb_path.parent)

    print("Executando células sequencialmente no PySpark...")
    execution_count = 0
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            execution_count += 1
            cell["execution_count"] = execution_count
            code = "".join(cell["source"])
            
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            
            print(f"Executando Célula de Código #{execution_count} (índice {idx})...")
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
                print(f"  -> Célula #{execution_count} executada com sucesso! Linhas de output: {len(output_text.splitlines())}")
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

    os.chdir(orig_cwd)

    print(f"Gravando notebook atualizado com outputs em: {nb_path}")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Validação do notebook gravado:")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_check = json.load(f)
    
    total_code = 0
    total_outputs = 0
    for idx, cell in enumerate(nb_check["cells"]):
        if cell["cell_type"] == "code":
            total_code += 1
            outs = len(cell.get("outputs", []))
            if outs > 0:
                total_outputs += 1
            print(f"  Célula {idx:02d} [CODE #{cell.get('execution_count')}]: outputs={outs}")
    
    print(f"\nFinalizado! {total_outputs}/{total_code} células de código possuem outputs salvos.")

if __name__ == "__main__":
    main()
