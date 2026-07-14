# Importa subprocess para ejecutar otros programas o scripts
# desde este archivo de Python.
import subprocess

# Importa sys para obtener información del intérprete
# de Python que está ejecutando actualmente el programa.
import sys

# Importa Path para trabajar con rutas de archivos
# de manera segura y compatible con diferentes sistemas operativos.
from pathlib import Path


# Obtiene la ruta absoluta de la carpeta donde se encuentra
# este archivo Python.
BASE_DIR = Path(__file__).resolve().parent


def run_command(command):
    """
    Ejecuta un comando externo dentro de la carpeta principal del proyecto.

    Parámetros:
        command (list): Lista que contiene el programa y sus argumentos.

    Ejemplo:
        [
            "C:/proyecto/.venv/Scripts/python.exe",
            "generate_metadata.py"
        ]

    Si el comando termina con un error, se genera una excepción.
    """

    # Imprime una línea en blanco para separar visualmente
    # la ejecución de cada comando.
    print()

    # Muestra en consola el comando que se va a ejecutar.
    # join convierte la lista del comando en una cadena de texto.
    print(f"Ejecutando: {' '.join(command)}")

    # Imprime una línea separadora de 50 guiones.
    print("-" * 50)

    # Ejecuta el comando indicado.
    result = subprocess.run(
        command,

        # Establece BASE_DIR como carpeta de trabajo.
        # Esto permite localizar correctamente los scripts
        # generate_metadata.py y build_index.py.
        cwd=BASE_DIR,

        # Indica que la entrada y salida del proceso
        # deben tratarse como texto.
        text=True
    )

    # Verifica el código de salida del comando.
    # Un código diferente de cero indica que ocurrió un error.
    if result.returncode != 0:

        # Detiene el programa y muestra cuál comando falló.
        raise RuntimeError(
            f"El comando falló: {' '.join(command)}"
        )


def main():
    """
    Ejecuta el proceso completo de actualización del índice:

    1. Genera el archivo metadata.json.
    2. Genera los embeddings y las etiquetas.
    3. Informa que el proceso terminó correctamente.
    """

    # Obtiene la ruta del intérprete de Python que está
    # ejecutando actualmente este archivo.
    #
    # Esto es importante cuando se utiliza un entorno virtual,
    # porque asegura que los demás scripts se ejecuten utilizando
    # el mismo entorno y las mismas librerías instaladas.
    python_executable = sys.executable

    # Ejecuta generate_metadata.py para analizar las carpetas
    # de personajes y actualizar el archivo metadata.json.
    run_command([
        python_executable,
        "generate_metadata.py"
    ])

    # Ejecuta build_index.py para procesar las imágenes
    # y actualizar los archivos embeddings.npy y labels.json.
    #
    # Este comando solo se ejecutará si generate_metadata.py
    # terminó correctamente.
    run_command([
        python_executable,
        "build_index.py"
    ])

    # Imprime una línea en blanco para separar el mensaje final.
    print()

    # Informa que todos los comandos terminaron correctamente.
    print("Proceso completado correctamente.")

    # Muestra los archivos que fueron actualizados durante el proceso.
    print(
        "metadata.json, embeddings.npy y labels.json fueron actualizados."
    )


# Comprueba si este archivo se está ejecutando directamente.
#
# Evita que main() se ejecute automáticamente cuando este archivo
# sea importado desde otro módulo de Python.
if __name__ == "__main__":

    # Inicia el proceso completo de actualización.
    main()