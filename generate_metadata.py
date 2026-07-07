import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGES_DIR = BASE_DIR / "reference_images"
METADATA_PATH = BASE_DIR / "metadata.json"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS


def generate_metadata():
    if not REFERENCE_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {REFERENCE_IMAGES_DIR}"
        )

    metadata = []

    character_folders = [
        folder
        for folder in REFERENCE_IMAGES_DIR.iterdir()
        if folder.is_dir()
    ]

    if len(character_folders) == 0:
        print("No se encontraron carpetas de personajes.")
        return

    for character_folder in sorted(character_folders):
        character_name = character_folder.name

        image_files = [
            file
            for file in character_folder.rglob("*")
            if file.is_file() and is_image_file(file)
        ]

        if len(image_files) == 0:
            print(f"Sin imágenes para: {character_name}")
            continue

        images = []

        for image_file in sorted(image_files):
            relative_path = image_file.relative_to(BASE_DIR)
            images.append(str(relative_path).replace("\\", "/"))

        metadata.append({
            "character": character_name,
            "images": images
        })

        print(f"{character_name}: {len(images)} imágenes encontradas.")

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print()
    print("metadata.json generado correctamente.")
    print(f"Total de personajes: {len(metadata)}")
    print(f"Ruta: {METADATA_PATH}")


if __name__ == "__main__":
    generate_metadata()