"""Prepare the offline science environment: pinned upstream code, Python packages and a hardware check.

Runs the same on Google Colab, a local GPU machine or CPU-only. Use Python 3.11 or 3.12 (upstream
requirement). Installs into the current interpreter, so activate your virtual environment first.

Usage: python -m python scripts/setup_environment.py [--skip-install]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from moonatlas_science import config


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def checkout_upstream() -> None:
    target = config.UPSTREAM_CODE_DIR
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", config.UPSTREAM_CODE["url"], str(target)])
    run(["git", "fetch", "--quiet", "origin"], cwd=target)
    run(["git", "checkout", "--quiet", config.UPSTREAM_CODE["commit"]], cwd=target)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=target, capture_output=True, text=True, check=True).stdout.strip()
    if head != config.UPSTREAM_CODE["commit"]:
        raise SystemExit(f"upstream checkout is at {head}, expected {config.UPSTREAM_CODE['commit']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-install", action="store_true", help="only check out upstream code and verify imports")
    args = parser.parse_args()

    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise SystemExit(f"Python {sys.version.split()[0]} found; upstream requires 3.11 or 3.12")
    checkout_upstream()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(config.UPSTREAM_CODE_DIR)])
        run([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(Path(__file__).with_name("requirements.txt"))])

    import terratorch  # noqa: F401,PLC0415
    import terratorch_integration  # noqa: F401,PLC0415
    import torch  # noqa: PLC0415

    print(f"torch {torch.__version__} · CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU: inference will run on CPU (slower, same outputs up to floating-point nondeterminism).")
    print("Environment ready.")


if __name__ == "__main__":
    main()
