import json

nb = json.load(open('notebooks/pipeline_mllib.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    first_line = c['source'][0].strip() if c['source'] else ''
    print(f"Cell {i:02d} [{c['cell_type']}]: {first_line[:80]}")
