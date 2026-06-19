"""
Exportar modelo D-FINE-seg para deploy (ONNX, TensorRT, OpenVINO, CoreML, LiteRT)

Análogo ao export.py do projeto YOLO. Executa src.dl.export do D-FINE-seg.

Uso:
    python export.py                      # todos os backends configurados
    python export.py --format onnx
    python export.py --format tensorrt --half
    python export.py --from-pretrained    # exporta pesos pretreinados sem treinar

Observação:
    TensorRT precisa ser exportado no mesmo device (GPU) onde será usado.
"""

import argparse

from utils.dfine_runner import run_module


FORMATS = ["onnx", "tensorrt", "openvino", "coreml", "litert"]


def parse_args():
    parser = argparse.ArgumentParser(description="Exportar modelo D-FINE-seg")

    parser.add_argument("--exp-name", default="exp", help="Experimento treinado a exportar")
    parser.add_argument("--task", default="segment", choices=["detect", "segment"])
    parser.add_argument(
        "--format",
        default=None,
        choices=FORMATS,
        help="Backend único (padrão: todos os configurados em config.yaml)",
    )
    parser.add_argument("--half", action="store_true", help="Exportar em FP16 (tensorrt/openvino)")
    parser.add_argument("--dynamic", action="store_true", help="Batch dinâmico (onnx/openvino)")
    parser.add_argument(
        "--from-pretrained",
        action="store_true",
        help="Exportar pesos pretreinados (dispensa treino)",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Overrides Hydra extras, ex.: export.max_batch_size=4",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    overrides = [
        f"task={args.task}",
        f"exp_name={args.exp_name}",
        f"export.half={'True' if args.half else 'False'}",
        f"export.dynamic_input={'True' if args.dynamic else 'False'}",
        f"export.from_pretrained={'True' if args.from_pretrained else 'False'}",
        *args.extra,
    ]
    if args.format:
        overrides.append(f"export.formats=[{args.format}]")

    print(f"\nExperimento: {args.exp_name}")
    print(f"Formato:     {args.format or 'todos (config.yaml)'}")
    print(f"FP16:        {args.half}\n")

    return run_module("src.dl.export", overrides)


if __name__ == "__main__":
    raise SystemExit(main())
