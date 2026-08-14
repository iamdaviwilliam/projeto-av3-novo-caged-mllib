"""
Script de Extração Automatizada dos Microdados do Novo CAGED.

Recursos e Regras:
- Busca arquivos .7z em data/raw/archives/
- Identifica tipo (CAGEDMOV, CAGEDEXC, CAGEDFOR) e competência (AAAAMM)
- Extrai dados para data/raw/extracted/{ano}/{competência}/
- Preserva arquivos originais .7z
- Não sobrescreve arquivos existentes sem aviso (salva status como IGNORADO)
- Permite filtro por competência via argumento --competencia (ex: 202412)
- Permite filtro por tipo via argumento --tipo (ex: CAGEDMOV)
"""

import argparse
import re
import sys
from pathlib import Path
import py7zr


def parse_filename(filename: str):
    """
    Analisa o nome do arquivo .7z para extrair tipo, ano, mês e competência.
    Exemplo: CAGEDMOV202412.7z -> tipo=CAGEDMOV, ano=2024, mês=12, competencia=202412
    """
    pattern = r"^(CAGEDMOV|CAGEDEXC|CAGEDFOR)(\d{4})(\d{2})\.7z$"
    match = re.match(pattern, filename, re.IGNORECASE)
    if not match:
        return None

    tipo = match.group(1).upper()
    ano = match.group(2)
    mes = match.group(3)
    competencia = f"{ano}{mes}"

    # Validação da competência
    ano_int = int(ano)
    mes_int = int(mes)
    if not (2000 <= ano_int <= 2100 and 1 <= mes_int <= 12):
        return None

    return {
        "tipo": tipo,
        "ano": ano,
        "mes": mes,
        "competencia": competencia
    }


def get_rel_path(path: Path) -> Path:
    """Retorna o caminho relativo à raiz de trabalho se possível."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path


def extract_caged(archives_dir: Path, extracted_dir: Path, target_competencia: str = None, target_tipo: str = None):
    archives_dir = Path(archives_dir)
    extracted_dir = Path(extracted_dir)

    if not archives_dir.exists():
        print(f"Erro: Diretório de arquivos '{archives_dir}' não encontrado.")
        sys.exit(1)

    all_archives = list(archives_dir.rglob("*.7z"))
    print(f"Total de arquivos .7z localizados em '{archives_dir}': {len(all_archives)}")

    found_list = []
    extracted_list = []
    ignored_list = []
    error_list = []

    for archive_path in sorted(all_archives):
        info = parse_filename(archive_path.name)
        if not info:
            print(f"[AVISO] Arquivo com formato não reconhecido: {get_rel_path(archive_path)}")
            continue

        comp = info["competencia"]
        ano = info["ano"]
        tipo = info["tipo"]

        # Filtro de competência
        if target_competencia and comp != target_competencia:
            continue

        # Filtro de tipo
        if target_tipo and tipo != target_tipo.upper():
            continue

        found_list.append((archive_path, info))

        out_folder = extracted_dir / ano / comp
        out_folder.mkdir(parents=True, exist_ok=True)

        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                target_names = archive.getnames()
                
                # Verificar se todos os arquivos internos já foram extraídos
                already_exists = all((out_folder / name).exists() for name in target_names)

                if already_exists:
                    ignored_list.append((archive_path, out_folder, "Arquivos já extraídos previamente"))
                    print(f"[IGNORADO] {archive_path.name} -> {get_rel_path(out_folder)} (já existe)")
                else:
                    archive.extractall(path=out_folder)
                    extracted_list.append((archive_path, out_folder, target_names))
                    print(f"[EXTRAÍDO] {archive_path.name} -> {get_rel_path(out_folder)}")

        except Exception as e:
            error_list.append((archive_path, str(e)))
            print(f"[ERRO] Falha ao extrair {archive_path.name}: {e}")

    print("\n" + "=" * 60)
    print("RESUMO DA OPERAÇÃO DE EXTRAÇÃO")
    print("=" * 60)
    if target_competencia:
        print(f"Filtro por Competência: {target_competencia}")
    if target_tipo:
        print(f"Filtro por Tipo: {target_tipo}")
    print(f"Encontrados matching: {len(found_list)}")
    print(f"Extraídos com sucesso: {len(extracted_list)}")
    print(f"Ignorados (já existentes): {len(ignored_list)}")
    print(f"Erros: {len(error_list)}")
    print("=" * 60)

    return {
        "found": found_list,
        "extracted": extracted_list,
        "ignored": ignored_list,
        "errors": error_list
    }


def main():
    parser = argparse.ArgumentParser(description="Script de extração automatizada de microdados do Novo CAGED (.7z)")
    parser.add_argument("--competencia", type=str, default=None, help="Competência específica no formato AAAAMM (ex: 202412)")
    parser.add_argument("--tipo", type=str, default=None, help="Tipo de arquivo (ex: CAGEDMOV, CAGEDEXC, CAGEDFOR)")
    parser.add_argument("--archives-dir", type=str, default="data/raw/archives", help="Diretório dos arquivos .7z originais")
    parser.add_argument("--extracted-dir", type=str, default="data/raw/extracted", help="Diretório de destino dos arquivos extraídos")

    args = parser.parse_args()
    extract_caged(Path(args.archives_dir), Path(args.extracted_dir), args.competencia, args.tipo)


if __name__ == "__main__":
    main()
