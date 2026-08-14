import os
import sys
import shutil
import json
import csv
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent

def get_dir_size(path):
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total

def main():
    print("=== AUDITORIA DA ESTRUTURA FÍSICA DO PROJETO ===")
    
    files_to_check = [
        "README.md",
        "data/README.md",
        "requirements.txt",
        "pyproject.toml",
        "uv.lock",
        ".gitignore",
        "notebooks/pipeline_mllib.ipynb",
        "silver/caged_nordeste_ml",
        "models/logistic_regression_nordeste",
        "models/random_forest_nordeste",
        "models/tuned_best_model_nordeste",
        "outputs/metrics/model_comparison_metrics.csv",
        "outputs/target_audit_nordeste/df_modelagem.parquet"
    ]

    for item in files_to_check:
        p = root_dir / item
        if p.exists():
            if p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  [OK] ARQUIVO: {item:45s} | {size_mb:.2f} MB")
            else:
                size_mb = get_dir_size(p) / (1024 * 1024)
                print(f"  [OK] PASTA:   {item:45s} | {size_mb:.2f} MB")
        else:
            print(f"  [AUSENTE]     {item:45s}")

if __name__ == "__main__":
    main()
