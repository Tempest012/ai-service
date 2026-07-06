import json
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor


BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
LABELS_PATH = BASE_DIR / "labels.json"

MODEL_NAME = "openai/clip-vit-base-patch32"

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.28"))


class CandidateResponse(BaseModel):
    character: str
    confidence: float


class RecognitionResponse(BaseModel):
    identified: bool
    character: Optional[str]
    confidence: float
    topCandidates: List[CandidateResponse]


app = FastAPI(
    title="Proyecto Personajes - AI Service",
    description="Servicio de reconocimiento visual de personajes usando CLIP.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model: CLIPModel | None = None
processor: CLIPProcessor | None = None
embeddings: np.ndarray | None = None
labels: list[dict] | None = None


@app.on_event("startup")
def startup_event():
    global model, processor, embeddings, labels

    print(f"Usando dispositivo: {device}")
    print("Cargando modelo CLIP...")

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    if not EMBEDDINGS_PATH.exists() or not LABELS_PATH.exists():
        print("No se encontraron embeddings.npy o labels.json.")
        print("Ejecuta primero: python build_index.py")
        embeddings = None
        labels = None
        return

    embeddings = np.load(EMBEDDINGS_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    print(f"Índice cargado con {len(labels)} imágenes de referencia.")


def encode_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    if model is None or processor is None:
        raise RuntimeError("El modelo CLIP no está cargado.")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

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


def search_character(query_embedding: np.ndarray):
    if embeddings is None or labels is None:
        raise RuntimeError("No hay índice cargado. Ejecuta build_index.py primero.")

    similarities = embeddings @ query_embedding

    best_by_character: dict[str, float] = {}

    for label, score in zip(labels, similarities):
        character_name = label["character"]
        score_float = float(score)

        if character_name not in best_by_character:
            best_by_character[character_name] = score_float
        else:
            best_by_character[character_name] = max(
                best_by_character[character_name],
                score_float
            )

    ordered_candidates = sorted(
        best_by_character.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_candidates = [
        CandidateResponse(
            character=character,
            confidence=max(0.0, min(1.0, score))
        )
        for character, score in ordered_candidates[:5]
    ]

    best_character, best_score = ordered_candidates[0]

    confidence = max(0.0, min(1.0, float(best_score)))
    identified = confidence >= SIMILARITY_THRESHOLD

    return RecognitionResponse(
        identified=identified,
        character=best_character if identified else None,
        confidence=confidence,
        topCandidates=top_candidates
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelLoaded": model is not None,
        "indexLoaded": embeddings is not None and labels is not None,
        "threshold": SIMILARITY_THRESHOLD
    }


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(image: UploadFile = File(...)):
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen."
        )

    image_bytes = await image.read()

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="La imagen está vacía."
        )

    try:
        query_embedding = encode_image_from_bytes(image_bytes)
        result = search_character(query_embedding)
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al reconocer personaje: {str(error)}"
        )