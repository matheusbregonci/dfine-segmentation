"""
Inferência com D-FINE-seg

Análogo ao predict.py do projeto YOLO. Executa o módulo de inferência do D-FINE-seg
(src.dl.infer), que roda o modelo treinado sobre uma pasta de imagens/vídeos e salva
as visualizações em output/infer/.

Uso:
    python predict.py
    python predict.py --source data/test --exp-name exp
    python predict.py --source minha_pasta/ --conf 0.5

Observações:
    - O D-FINE-seg seleciona os pesos a partir do experimento treinado (exp-name);
      use o mesmo --exp-name usado no treino.
    - --source aponta para a pasta de teste (train.path_to_test_data).
"""

import argparse

from utils.dfine_runner import run_module


def parse_args():
    parser = argparse.ArgumentParser(description="Inferência D-FINE-seg segmentação")

    parser.add_argument("--source", default="data/test", help="Pasta de imagens/vídeos")
    parser.add_argument("--exp-name", default="exp", help="Experimento treinado a usar")
    parser.add_argument("--task", default="segment", choices=["detect", "segment"])
    parser.add_argument("--conf", type=float, default=0.5, help="Confiança mínima (0-1)")
    parser.add_argument("--iou", type=float, default=0.2, help="Limiar IoU")
    parser.add_argument("--device", default="cuda", help="cuda ou cpu")
    parser.add_argument("--no-crop", action="store_true", help="Não salvar crops dos objetos")
    parser.add_argument(
        "extra",
        nargs="*",
        help="Overrides Hydra extras, ex.: infer.to_track=False",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    overrides = [
        f"task={args.task}",
        f"exp_name={args.exp_name}",
        f"train.path_to_test_data={args.source}",
        f"train.conf_thresh={args.conf}",
        f"train.iou_thresh={args.iou}",
        f"train.device={args.device}",
        f"infer.to_crop={'False' if args.no_crop else 'True'}",
        *args.extra,
    ]

    print(f"\nFonte:       {args.source}")
    print(f"Experimento: {args.exp_name}")
    print(f"Confiança:   {args.conf} | IoU: {args.iou}\n")

    return run_module("src.dl.infer", overrides)


if __name__ == "__main__":
    raise SystemExit(main())
