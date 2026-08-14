from pathlib import Path
import json

nb_path = Path("notebooks/pipeline_mllib.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        outs = cell.get("outputs", [])
        if "AUDITORIA DE NULOS" in src or "col_null_expr" in src:
            print(f"=== Code Cell Index {idx} ===")
            print(f"Outputs count: {len(outs)}")
            for o in outs:
                print("--- Text Output ---")
                print("".join(o.get("text", [])))
