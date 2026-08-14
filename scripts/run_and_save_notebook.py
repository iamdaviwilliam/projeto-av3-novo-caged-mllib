import os
import json
import io
import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'scripts'))
import build_notebook as bn

nb_path = root_dir / 'notebooks' / 'pipeline_mllib.ipynb'

def main():
    hadoop_home = (root_dir / 'hadoop').resolve()
    os.environ['HADOOP_HOME'] = str(hadoop_home)
    os.environ['PATH'] = str(hadoop_home / 'bin') + os.pathsep + os.environ.get('PATH', '')

    print("=== 1. Obtendo estrutura do Notebook ===")
    nb = bn.notebook_json

    exec_globals = {}
    orig_cwd = os.getcwd()
    os.chdir(nb_path.parent)

    print("=== 2. Executando células e capturando outputs ===")
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
                print(f"  -> Sucesso! Linhas de output: {len(output_text.splitlines())}")
            except Exception as e:
                print(f"  -> ERRO na Célula #{execution_count}: {e}")
                cell["outputs"] = [{
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "output_type": "error",
                    "traceback": [str(e)]
                }]
                break

    os.chdir(orig_cwd)

    print(f"\n=== 3. Salvando Notebook Atomicamente em: {nb_path} ===")
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("=== 4. Validação do arquivo gravado ===")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_check = json.load(f)

    code_cells = [c for c in nb_check['cells'] if c['cell_type'] == 'code']
    cells_with_out = [c for c in code_cells if len(c.get('outputs', [])) > 0]
    print(f"Total de Células de Código: {len(code_cells)}")
    print(f"Células com Outputs Salvos: {len(cells_with_out)}/{len(code_cells)}")

if __name__ == "__main__":
    main()
