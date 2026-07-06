import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "metadata.json"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
LABELS_PATH = BASE_DIR / "labels.json"

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Usando dispositivo: {device}")
    print("Cargando modelo CLIP...")

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    return model, processor, device


def encode_image(image_path: Path, model, processor, device):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        features = model.get_image_features(pixel_values=pixel_values)

        if not isinstance(features, torch.Tensor):
            if hasattr(features, "image_embeds") and features.image_embeds is not None:
                features = features.image_embeds
            elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                features = features.pooler_output
            else:
                raise RuntimeError(
                    f"No se pudo obtener el embedding de la imagen. Tipo recibido: {type(features)}"
                )

    features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features.cpu().numpy()[0].astype("float32")


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError("No existe metadata.json")

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model, processor, device = load_model()

    embeddings = []
    labels = []

    for item in metadata:
        character_name = item.get("character")
        images = item.get("images", [])

        if not character_name:
            print("Registro sin nombre de personaje. Se omite.")
            continue

        for relative_image_path in images:
            image_path = BASE_DIR / relative_image_path

            if not image_path.exists():
                print(f"No existe la imagen: {image_path}")
                continue

            print(f"Procesando: {character_name} -> {relative_image_path}")

            embedding = encode_image(image_path, model, processor, device)

            embeddings.append(embedding)

            labels.append({
                "character": character_name,
                "sourceImage": relative_image_path
            })

    if len(embeddings) == 0:
        raise ValueError("No se generó ningún embedding. Revisa metadata.json e imágenes.")

    embeddings_array = np.vstack(embeddings)

    np.save(EMBEDDINGS_PATH, embeddings_array)

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(labels, file, ensure_ascii=False, indent=2)

    print("Índice generado correctamente.")
    print(f"Total de imágenes indexadas: {len(labels)}")
    print(f"Archivo generado: {EMBEDDINGS_PATH}")
    print(f"Archivo generado: {LABELS_PATH}")


if __name__ == "__main__":
    main()