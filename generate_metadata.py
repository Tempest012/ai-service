# Importa el módulo json para crear y guardar archivos en formato JSON.
import json

# Importa Path para trabajar con rutas y archivos de manera compatible
# con diferentes sistemas operativos.
from pathlib import Path


# Obtiene la ruta absoluta de la carpeta donde se encuentra este archivo Python.
BASE_DIR = Path(__file__).resolve().parent

# Define la ruta de la carpeta que contiene las imágenes de referencia.
REFERENCE_IMAGES_DIR = BASE_DIR / "reference_images"

# Define la ruta donde se creará el archivo metadata.json.
METADATA_PATH = BASE_DIR / "metadata.json"

# Conjunto de extensiones de imágenes permitidas.
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def is_image_file(file_path: Path) -> bool:
    """
    Verifica si un archivo tiene una extensión de imagen permitida.

    Parámetros:
        file_path (Path): Ruta del archivo que se desea comprobar.

    Retorna:
        bool: True si el archivo es una imagen permitida.
              False si su extensión no está permitida.
    """

    # Obtiene la extensión del archivo, la convierte a minúsculas
    # y comprueba si se encuentra en ALLOWED_EXTENSIONS.
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS


def generate_metadata():
    """
    Recorre las carpetas de personajes dentro de reference_images
    y genera automáticamente el archivo metadata.json.

    Cada carpeta representa un personaje y contiene sus imágenes
    de referencia.
    """

    # Verifica que exista la carpeta principal de imágenes.
    if not REFERENCE_IMAGES_DIR.exists():
        # Detiene el programa si la carpeta no existe.
        raise FileNotFoundError(
            f"No existe la carpeta: {REFERENCE_IMAGES_DIR}"
        )

    # Lista donde se almacenará la información de todos los personajes.
    metadata = []

    # Obtiene únicamente las carpetas que están dentro de reference_images.
    # Cada carpeta encontrada se considera un personaje.
    character_folders = [
        folder
        for folder in REFERENCE_IMAGES_DIR.iterdir()
        if folder.is_dir()
    ]

    # Comprueba si se encontraron carpetas de personajes.
    if len(character_folders) == 0:
        print("No se encontraron carpetas de personajes.")
        return

    # Recorre las carpetas de personajes en orden alfabético.
    for character_folder in sorted(character_folders):

        # Utiliza el nombre de la carpeta como nombre o identificador
        # del personaje.
        character_name = character_folder.name

        # Busca de forma recursiva todos los archivos dentro de la carpeta
        # del personaje y conserva únicamente las imágenes permitidas.
        image_files = [
            file
            for file in character_folder.rglob("*")
            if file.is_file() and is_image_file(file)
        ]

        # Si la carpeta no contiene imágenes, muestra una advertencia
        # y continúa con el siguiente personaje.
        if len(image_files) == 0:
            print(f"Sin imágenes para: {character_name}")
            continue

        # Lista donde se guardarán las rutas de las imágenes del personaje.
        images = []

        # Recorre las imágenes encontradas en orden alfabético.
        for image_file in sorted(image_files):

            # Convierte la ruta absoluta de la imagen en una ruta relativa
            # respecto a la carpeta principal del proyecto.
            relative_path = image_file.relative_to(BASE_DIR)

            # Convierte la ruta a texto y reemplaza las diagonales invertidas
            # de Windows por diagonales normales para mantener un formato
            # compatible dentro del archivo JSON.
            images.append(str(relative_path).replace("\\", "/"))

        # Agrega al metadata el nombre del personaje y sus imágenes.
        metadata.append({
            "character": character_name,
            "images": images
        })

        # Muestra en la consola cuántas imágenes fueron encontradas
        # para el personaje actual.
        print(f"{character_name}: {len(images)} imágenes encontradas.")

    # Abre o crea el archivo metadata.json en modo escritura.
    with open(METADATA_PATH, "w", encoding="utf-8") as file:

        # Guarda la lista metadata dentro del archivo JSON.
        #
        # ensure_ascii=False:
        # Permite guardar correctamente caracteres como tildes y la letra ñ.
        #
        # indent=2:
        # Organiza el archivo JSON con una indentación de dos espacios.
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    # Muestra un espacio en blanco para separar los resultados.
    print()

    # Informa que el archivo se generó correctamente.
    print("metadata.json generado correctamente.")

    # Muestra el número total de personajes registrados.
    print(f"Total de personajes: {len(metadata)}")

    # Muestra la ruta completa donde se guardó el archivo.
    print(f"Ruta: {METADATA_PATH}")


# Comprueba que este archivo se esté ejecutando directamente.
# Evita que generate_metadata() se ejecute automáticamente
# cuando el archivo sea importado desde otro módulo.
if __name__ == "__main__":

    # Ejecuta la función que genera el archivo metadata.json.
    generate_metadata()