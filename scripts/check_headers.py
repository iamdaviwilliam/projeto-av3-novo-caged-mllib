from pathlib import Path

files = sorted(list(Path('data/raw/extracted').rglob('CAGEDMOV*.txt')))
with open(files[0], 'r', encoding='utf-8', errors='ignore') as fp:
    h = fp.readline().strip()
    cols = [c.strip() for c in h.split(';')]
    print("Columns count:", len(cols))
    for i, c in enumerate(cols):
        print(f"  {i:02d}: {c}")
