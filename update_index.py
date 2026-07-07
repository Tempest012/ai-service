import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_command(command):
    print()
    print(f"Ejecutando: {' '.join(command)}")
    print("-" * 50)

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"El comando falló: {' '.join(command)}"
        )


def main():
    python_executable = sys.executable

    run_command([python_executable, "generate_metadata.py"])
    run_command([python_executable, "build_index.py"])

    print()
    print("Proceso completado correctamente.")
    print("metadata.json, embeddings.npy y labels.json fueron actualizados.")


if __name__ == "__main__":
    main()