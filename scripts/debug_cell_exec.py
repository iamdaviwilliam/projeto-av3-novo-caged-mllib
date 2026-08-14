import os
import json
import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import sys

root_dir = Path('.').resolve()
sys.path.insert(0, str(root_dir / 'scripts'))
import build_notebook as bn

nb = bn.notebook_json

exec_globals = {}
os.chdir(root_dir / 'notebooks')

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = "".join(cell["source"])
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, exec_globals)
            out_txt = stdout_buf.getvalue()
            err_txt = stderr_buf.getvalue()
            if 'AUDITORIA DE NULOS' in code:
                print(f"=== CELL INDEX {idx} EXECUTED ===")
                print("STDOUT LEN:", len(out_txt))
                print("STDOUT CONTENT:\n", out_txt)
                print("STDERR CONTENT:\n", err_txt)
        except Exception as e:
            print(f"ERROR AT CELL {idx}: {e}")
            break
