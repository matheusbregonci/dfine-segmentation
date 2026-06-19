"""
Camada de ligação com o repositório D-FINE-seg.

O D-FINE-seg (https://github.com/ArgoHA/D-FINE-seg) não é um pacote pip como o
ultralytics: é um repositório que se clona e executa via `uv run python -m src.dl.*`,
configurado por um `config.yaml` (Hydra). Estes utilitários localizam esse clone,
sincronizam o nosso config e disparam os módulos com overrides de linha de comando.
"""

import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_dfine_home() -> Path:
    """Localiza o clone do D-FINE-seg.

    Ordem de busca: variável de ambiente DFINE_HOME, ./D-FINE-seg, ../D-FINE-seg.
    """
    candidates = []
    env = os.environ.get("DFINE_HOME")
    if env:
        candidates.append(Path(env))
    candidates += [
        PROJECT_ROOT / "D-FINE-seg",
        PROJECT_ROOT.parent / "D-FINE-seg",
    ]

    for c in candidates:
        if (c / "src" / "dl" / "train.py").exists():
            return c

    raise SystemExit(
        "D-FINE-seg não encontrado.\n"
        "  - Rode 'powershell scripts/setup_dfine.ps1' para clonar e instalar, ou\n"
        "  - Defina a variável de ambiente DFINE_HOME apontando para o clone."
    )


def sync_config(dfine_home: Path) -> Path:
    """Copia nosso config/config.yaml para a raiz do repo D-FINE-seg (onde o Hydra o lê)."""
    src = PROJECT_ROOT / "config" / "config.yaml"
    dst = dfine_home / "config.yaml"
    shutil.copyfile(src, dst)
    return dst


def run_module(module: str, overrides=None) -> int:
    """Executa `uv run python -m <module>` dentro do D-FINE-seg com overrides Hydra.

    `train.root` é sempre forçado para esta pasta para que dataset e saídas fiquem aqui.
    """
    overrides = list(overrides or [])
    dfine_home = find_dfine_home()
    sync_config(dfine_home)

    overrides.append(f"train.root={PROJECT_ROOT.as_posix()}")

    cmd = ["uv", "run", "python", "-m", module] + overrides
    print("D-FINE-seg:", dfine_home)
    print("Comando:   ", " ".join(cmd), "\n")

    return subprocess.run(cmd, cwd=str(dfine_home)).returncode
