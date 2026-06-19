"""
Preparação do dataset para o D-FINE-seg.

O D-FINE-seg (formato YOLO) espera UMA pasta com todas as imagens e labels:

    data/dataset/
    ├── images/   (.jpg/.png)
    └── labels/   (.txt — polígonos normalizados: classe x1 y1 x2 y2 ... xN yN)

e gera o split train/val internamente com `python -m src.etl.split`.

O projeto YOLO vizinho usa o dataset já dividido em train/val/test. Este script
consolida aqueles splits na pasta data/dataset/ (reaproveitando o MESMO dataset gran,
pois o formato de labels é idêntico) e copia o split de teste para data/test/ para
inferência.

Uso:
    python utils/prepare_data.py --src ../yolov8-segmentation/data
    python utils/prepare_data.py --src ../yolov8-segmentation/data --link   # symlink ao invés de copiar
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_GLOB = "*.[jp][pn][ge]*"


def _transfer(src_file: Path, dst_file: Path, link: bool):
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if dst_file.exists():
        return
    if link:
        try:
            dst_file.symlink_to(src_file)
            return
        except OSError:
            pass  # sem privilégio de symlink no Windows -> copia
    shutil.copy2(src_file, dst_file)


def prepare(src_root: Path, link: bool):
    dataset_imgs = PROJECT_ROOT / "data" / "dataset" / "images"
    dataset_lbls = PROJECT_ROOT / "data" / "dataset" / "labels"
    test_imgs = PROJECT_ROOT / "data" / "test"

    n_imgs = n_lbls = n_test = 0

    for split in ["train", "val", "test"]:
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if not img_dir.exists():
            print(f"  (pulando '{split}': {img_dir} não existe)")
            continue

        for img in img_dir.glob(IMG_GLOB):
            _transfer(img, dataset_imgs / img.name, link)
            n_imgs += 1

            lbl = lbl_dir / img.with_suffix(".txt").name
            if lbl.exists():
                _transfer(lbl, dataset_lbls / lbl.name, link)
                n_lbls += 1

            # o split de teste também alimenta a pasta de inferência
            if split == "test":
                _transfer(img, test_imgs / img.name, link)
                n_test += 1

    print("\nDataset preparado para D-FINE-seg:")
    print(f"  data/dataset/images : {n_imgs} imagens")
    print(f"  data/dataset/labels : {n_lbls} labels")
    print(f"  data/test           : {n_test} imagens (inferência)")
    print("\nGere o split train/val com (dentro do clone do D-FINE-seg):")
    print("  uv run python -m src.etl.split")


def main():
    parser = argparse.ArgumentParser(description="Preparar dataset para D-FINE-seg")
    parser.add_argument(
        "--src",
        default="../yolov8-segmentation/data",
        help="Raiz do dataset YOLO (com train/val/test)",
    )
    parser.add_argument("--link", action="store_true", help="Usar symlinks ao invés de copiar")
    args = parser.parse_args()

    src_root = (PROJECT_ROOT / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src)
    if not src_root.exists():
        raise SystemExit(f"Origem não encontrada: {src_root}")

    print(f"Origem: {src_root}")
    prepare(src_root, args.link)


if __name__ == "__main__":
    main()
