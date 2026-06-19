"""
Treinamento D-FINE-seg - Segmentação de Instâncias

Análogo ao train.py do projeto YOLO, mas executando o framework D-FINE-seg
(https://github.com/ArgoHA/D-FINE-seg) por baixo.

Uso:
    python train.py
    python train.py --model l --epochs 200 --batch 8
    python train.py --exp-name run2 -- train.use_wandb=True   # overrides Hydra extras
"""

import argparse

from utils.dfine_runner import run_module


def parse_args():
    parser = argparse.ArgumentParser(description="Treinar D-FINE-seg para segmentação")

    parser.add_argument(
        "--model",
        default="s",
        choices=["n", "s", "m", "l", "x"],
        help="Tamanho do modelo D-FINE (n/s/m/l/x)",
    )
    parser.add_argument("--task", default="segment", choices=["detect", "segment"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4, help="-1 = auto")
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--device", default="cuda", help="cuda ou cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exp-name", default="exp")
    parser.add_argument(
        "extra",
        nargs="*",
        help="Overrides Hydra extras, ex.: train.use_ema=False train.early_stopping=0",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    overrides = [
        f"model_name={args.model}",
        f"task={args.task}",
        f"exp_name={args.exp_name}",
        f"train.epochs={args.epochs}",
        f"train.batch_size={args.batch}",
        f"train.img_size=[{args.imgsz},{args.imgsz}]",
        f"train.device={args.device}",
        f"train.num_workers={args.workers}",
        *args.extra,
    ]

    print(f"\nModelo:      dfine_{args.model}")
    print(f"Tarefa:      {args.task}")
    print(f"Épocas:      {args.epochs}")
    print(f"Batch:       {args.batch}")
    print(f"Imagem:      {args.imgsz}x{args.imgsz}")
    print(f"Dispositivo: {args.device}\n")

    return run_module("src.dl.train", overrides)


if __name__ == "__main__":
    raise SystemExit(main())
