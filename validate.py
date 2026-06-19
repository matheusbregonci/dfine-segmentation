"""
Avaliação / benchmark do modelo treinado (D-FINE-seg)

Análogo ao validate.py do projeto YOLO. No D-FINE-seg a validação acontece a cada
época durante o treino (métricas mAP/F1/IoU sobre o split de validação). Para uma
avaliação separada dos pesos treinados sobre o conjunto de validação, este script
chama o módulo de benchmark (src.dl.bench), que mede métricas e latência por backend.

Uso:
    python validate.py
    python validate.py --exp-name run2

Dica:
    Para inspecionar falsos positivos/negativos contra o ground truth, use:
        python -m src.dl.check_errors   (dentro do clone do D-FINE-seg)
"""

import argparse

from utils.dfine_runner import run_module


def parse_args():
    parser = argparse.ArgumentParser(description="Validar/benchmark D-FINE-seg")

    parser.add_argument("--exp-name", default="exp", help="Experimento treinado a avaliar")
    parser.add_argument("--task", default="segment", choices=["detect", "segment"])
    parser.add_argument("--device", default="cuda", help="cuda ou cpu")
    parser.add_argument(
        "extra",
        nargs="*",
        help="Overrides Hydra extras, ex.: bench.formats=[torch]",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    overrides = [
        f"task={args.task}",
        f"exp_name={args.exp_name}",
        f"train.device={args.device}",
        *args.extra,
    ]

    print(f"\nExperimento: {args.exp_name}")
    print(f"Tarefa:      {args.task}\n")

    return run_module("src.dl.bench", overrides)


if __name__ == "__main__":
    raise SystemExit(main())
