# Importa json para leer el archivo labels.json.
import json

# Importa os para acceder a variables de entorno del sistema.
import os

# Importa BytesIO para convertir los bytes recibidos
# desde la API en un objeto que pueda leer Pillow.
from io import BytesIO

# Importa Path para trabajar con rutas de archivos
# de forma segura y compatible con diferentes sistemas operativos.
from pathlib import Path

# Importa tipos utilizados en los modelos de respuesta.
from typing import List, Optional

# Importa NumPy para trabajar con los embeddings
# y calcular las similitudes entre imágenes.
import numpy as np

# Importa PyTorch, utilizado para ejecutar el modelo CLIP.
import torch

# Importa las herramientas principales de FastAPI.
#
# FastAPI:
# Crea la aplicación web.
#
# File:
# Indica que un parámetro corresponde a un archivo.
#
# HTTPException:
# Permite devolver errores HTTP personalizados.
#
# UploadFile:
# Representa el archivo enviado por el usuario.
from fastapi import FastAPI, File, HTTPException, UploadFile

# Importa el middleware CORS para permitir solicitudes
# provenientes de otras aplicaciones, como Angular o .NET.
from fastapi.middleware.cors import CORSMiddleware

# Importa Image de Pillow para abrir y convertir imágenes.
from PIL import Image

# Importa BaseModel para crear modelos de datos
# y respuestas mediante Pydantic.
from pydantic import BaseModel

# Importa el modelo CLIP y su procesador.
from transformers import CLIPModel, CLIPProcessor


# Obtiene la ruta absoluta de la carpeta
# donde se encuentra este archivo.
BASE_DIR = Path(__file__).resolve().parent

# Define la ruta del archivo que contiene
# los embeddings de las imágenes de referencia.
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"

# Define la ruta del archivo que contiene
# las etiquetas relacionadas con cada embedding.
LABELS_PATH = BASE_DIR / "labels.json"

# Nombre del modelo CLIP preentrenado
# que será cargado desde Hugging Face.
MODEL_NAME = "openai/clip-vit-base-patch32"

# Obtiene el umbral de similitud desde una variable de entorno.
#
# Si la variable SIMILARITY_THRESHOLD no existe,
# se utiliza 0.28 como valor predeterminado.
#
# Una imagen será considerada identificada cuando
# su similitud sea igual o superior a este valor.
SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", "0.28")
)


class CandidateResponse(BaseModel):
    """
    Representa uno de los personajes candidatos
    encontrados durante el reconocimiento.
    """

    # Nombre del personaje candidato.
    character: str

    # Nivel de similitud o confianza del candidato.
    confidence: float


class RecognitionResponse(BaseModel):
    """
    Representa la respuesta completa del servicio
    de reconocimiento de personajes.
    """

    # Indica si el personaje superó el umbral de similitud.
    identified: bool

    # Nombre del personaje identificado.
    #
    # Puede ser None si ningún personaje supera el umbral.
    character: Optional[str]

    # Nivel de confianza del mejor resultado.
    confidence: float

    # Lista con los mejores personajes candidatos.
    topCandidates: List[CandidateResponse]


# Crea la aplicación principal de FastAPI.
app = FastAPI(
    # Nombre que aparecerá en la documentación de Swagger.
    title="Proyecto Personajes - AI Service",

    # Descripción general del servicio.
    description=(
        "Servicio de reconocimiento visual "
        "de personajes usando CLIP."
    ),

    # Versión actual de la API.
    version="1.0.0"
)


# Agrega el middleware CORS a la aplicación.
#
# Esto permite que otras aplicaciones, como Angular
# o una API de .NET, puedan enviar solicitudes.
app.add_middleware(
    CORSMiddleware,

    # Permite solicitudes desde cualquier origen.
    #
    # En producción es recomendable reemplazar "*"
    # por las direcciones específicas permitidas.
    allow_origins=["*"],

    # Permite el envío de credenciales.
    allow_credentials=True,

    # Permite todos los métodos HTTP:
    # GET, POST, PUT, DELETE, entre otros.
    allow_methods=["*"],

    # Permite todos los encabezados HTTP.
    allow_headers=["*"],
)


# Selecciona el dispositivo donde se ejecutará el modelo.
#
# Utiliza CUDA si existe una GPU compatible.
# En caso contrario, utiliza el procesador.
device = "cuda" if torch.cuda.is_available() else "cpu"


# Variables globales que se inicializarán
# cuando la aplicación FastAPI inicie.
#
# Al principio contienen None porque los archivos
# y el modelo todavía no han sido cargados.
model: CLIPModel | None = None
processor: CLIPProcessor | None = None
embeddings: np.ndarray | None = None
labels: list[dict] | None = None


@app.on_event("startup")
def startup_event():
    """
    Se ejecuta automáticamente cuando inicia la API.

    Su función es:

    1. Cargar el modelo CLIP.
    2. Cargar el procesador de imágenes.
    3. Cargar embeddings.npy.
    4. Cargar labels.json.
    """

    # Indica que se modificarán las variables globales
    # declaradas anteriormente.
    global model, processor, embeddings, labels

    # Muestra si se utilizará CPU o GPU.
    print(f"Usando dispositivo: {device}")

    # Informa que comenzará la carga del modelo.
    print("Cargando modelo CLIP...")

    # Descarga o carga desde la memoria caché
    # el modelo CLIP preentrenado.
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Carga el procesador correspondiente al modelo.
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Mueve el modelo al dispositivo seleccionado.
    model.to(device)

    # Coloca el modelo en modo evaluación.
    #
    # Esto desactiva funciones utilizadas solamente
    # durante el entrenamiento.
    model.eval()

    # Verifica que existan los archivos necesarios
    # para realizar las comparaciones.
    if (
        not EMBEDDINGS_PATH.exists()
        or not LABELS_PATH.exists()
    ):
        # Muestra instrucciones si los archivos
        # todavía no han sido generados.
        print(
            "No se encontraron embeddings.npy "
            "o labels.json."
        )
        print(
            "Ejecuta primero: python build_index.py"
        )

        # Mantiene el índice como no cargado.
        embeddings = None
        labels = None

        # Termina el evento de inicio sin detener la API.
        return

    # Carga la matriz de embeddings desde el archivo NumPy.
    embeddings = np.load(EMBEDDINGS_PATH)

    # Abre labels.json en modo lectura.
    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        # Convierte el contenido JSON en una lista de Python.
        labels = json.load(file)

    # Muestra la cantidad de imágenes disponibles
    # en el índice de reconocimiento.
    print(
        f"Índice cargado con {len(labels)} "
        "imágenes de referencia."
    )


def encode_image_from_bytes(
    image_bytes: bytes
) -> np.ndarray:
    """
    Convierte los bytes de una imagen en un embedding.

    Parámetros:
        image_bytes:
            Contenido binario de la imagen recibida
            mediante la API.

    Retorna:
        np.ndarray:
            Vector numérico normalizado de tipo float32.
    """

    # Verifica que el modelo y el procesador
    # hayan sido cargados correctamente.
    if model is None or processor is None:
        raise RuntimeError(
            "El modelo CLIP no está cargado."
        )

    # Convierte los bytes en un archivo temporal en memoria,
    # abre la imagen y la transforma al formato RGB.
    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    # Prepara la imagen para ser procesada por CLIP.
    inputs = processor(
        images=image,

        # Indica que los resultados deben devolverse
        # como tensores de PyTorch.
        return_tensors="pt"
    )

    # Obtiene los valores de píxeles procesados
    # y los mueve al dispositivo seleccionado.
    pixel_values = inputs["pixel_values"].to(device)

    # Desactiva el cálculo de gradientes.
    #
    # Esto reduce el consumo de memoria porque el modelo
    # solamente se utiliza para realizar predicciones.
    with torch.no_grad():

        # Genera el embedding visual de la imagen.
        features = model.get_image_features(
            pixel_values=pixel_values
        )

        # Algunas versiones de Transformers pueden devolver
        # un objeto en lugar de un tensor directamente.
        if not isinstance(features, torch.Tensor):

            # Intenta obtener el resultado desde image_embeds.
            if (
                hasattr(features, "image_embeds")
                and features.image_embeds is not None
            ):
                features = features.image_embeds

            # Si image_embeds no está disponible,
            # intenta utilizar pooler_output.
            elif (
                hasattr(features, "pooler_output")
                and features.pooler_output is not None
            ):
                features = features.pooler_output

            # Si no se encuentra un tensor válido,
            # se genera una excepción.
            else:
                raise RuntimeError(
                    "No se pudo obtener el embedding "
                    "de la imagen. "
                    f"Tipo recibido: {type(features)}"
                )

    # Normaliza el vector utilizando la norma L2.
    #
    # Esto permite comparar los embeddings
    # mediante similitud coseno.
    features = features / features.norm(
        p=2,
        dim=-1,
        keepdim=True
    )

    # Mueve el resultado a la CPU, lo convierte a NumPy,
    # toma el primer elemento del lote y utiliza float32.
    return (
        features
        .cpu()
        .numpy()[0]
        .astype("float32")
    )


def search_character(
    query_embedding: np.ndarray
):
    """
    Busca el personaje más parecido a la imagen recibida.

    Parámetros:
        query_embedding:
            Vector numérico generado para la imagen enviada
            por el usuario.

    Retorna:
        RecognitionResponse:
            Resultado del reconocimiento y sus candidatos.
    """

    # Verifica que el índice de imágenes
    # se encuentre cargado.
    if embeddings is None or labels is None:
        raise RuntimeError(
            "No hay índice cargado. "
            "Ejecuta build_index.py primero."
        )

    # Calcula la similitud entre la imagen recibida
    # y todas las imágenes de referencia.
    #
    # Como los embeddings están normalizados,
    # este producto matricial equivale a calcular
    # la similitud coseno.
    similarities = embeddings @ query_embedding

    # Diccionario que almacenará la mejor similitud
    # obtenida para cada personaje.
    best_by_character: dict[str, float] = {}

    # Recorre las etiquetas junto con sus similitudes.
    for label, score in zip(labels, similarities):

        # Obtiene el nombre del personaje.
        character_name = label["character"]

        # Convierte el valor de NumPy a float de Python.
        score_float = float(score)

        # Si el personaje todavía no se ha registrado,
        # guarda su primera puntuación.
        if character_name not in best_by_character:
            best_by_character[character_name] = score_float

        else:
            # Si el personaje tiene varias imágenes,
            # conserva solamente su puntuación más alta.
            best_by_character[character_name] = max(
                best_by_character[character_name],
                score_float
            )

    # Ordena los personajes desde la similitud más alta
    # hasta la similitud más baja.
    ordered_candidates = sorted(
        best_by_character.items(),

        # Ordena utilizando el valor de similitud.
        key=lambda item: item[1],

        # Coloca primero los valores más altos.
        reverse=True
    )

    # Crea la lista con los cinco mejores candidatos.
    top_candidates = [
        CandidateResponse(
            # Nombre del personaje candidato.
            character=character,

            # Limita el valor de confianza al rango de 0 a 1.
            confidence=max(
                0.0,
                min(1.0, score)
            )
        )
        for character, score in ordered_candidates[:5]
    ]

    # Obtiene el personaje con la similitud más alta.
    best_character, best_score = ordered_candidates[0]

    # Limita la confianza final al rango de 0 a 1.
    confidence = max(
        0.0,
        min(1.0, float(best_score))
    )

    # Determina si el personaje puede considerarse identificado.
    #
    # El resultado será verdadero cuando la confianza
    # sea igual o superior al umbral configurado.
    identified = (
        confidence >= SIMILARITY_THRESHOLD
    )

    # Construye y retorna la respuesta final.
    return RecognitionResponse(
        # Indica si la imagen fue identificada.
        identified=identified,

        # Retorna el nombre solamente si supera el umbral.
        character=(
            best_character
            if identified
            else None
        ),

        # Retorna la confianza del mejor candidato.
        confidence=confidence,

        # Retorna los cinco mejores candidatos.
        topCandidates=top_candidates
    )


@app.get("/health")
def health():
    """
    Endpoint que permite comprobar el estado de la API.

    Retorna información sobre:

    - Estado general del servicio.
    - Carga del modelo.
    - Carga del índice.
    - Umbral de similitud utilizado.
    """

    return {
        # Indica que la API está activa.
        "status": "ok",

        # Indica si el modelo CLIP fue cargado.
        "modelLoaded": model is not None,

        # Indica si embeddings y etiquetas
        # están disponibles.
        "indexLoaded": (
            embeddings is not None
            and labels is not None
        ),

        # Muestra el umbral configurado.
        "threshold": SIMILARITY_THRESHOLD
    }


@app.post(
    "/recognize",
    response_model=RecognitionResponse
)
async def recognize(
    image: UploadFile = File(...)
):
    """
    Recibe una imagen y busca el personaje más parecido.

    La imagen debe enviarse mediante una solicitud
    multipart/form-data con el nombre de campo "image".
    """

    # Verifica que el archivo recibido tenga
    # un tipo de contenido correspondiente a una imagen.
    if (
        image.content_type is None
        or not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            # Código HTTP de solicitud incorrecta.
            status_code=400,

            # Mensaje que recibirá el cliente.
            detail="El archivo debe ser una imagen."
        )

    # Lee de forma asíncrona todos los bytes
    # del archivo enviado.
    image_bytes = await image.read()

    # Verifica que el archivo contenga información.
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="La imagen está vacía."
        )

    try:
        # Convierte la imagen recibida en un embedding.
        query_embedding = encode_image_from_bytes(
            image_bytes
        )

        # Busca el personaje más parecido
        # dentro del índice.
        result = search_character(
            query_embedding
        )

        # Devuelve el resultado al cliente.
        return result

    except Exception as error:
        # Captura cualquier error producido durante
        # la lectura, procesamiento o comparación.
        raise HTTPException(
            # Indica un error interno del servidor.
            status_code=500,

            # Devuelve una descripción del error.
            detail=(
                "Error al reconocer personaje: "
                f"{str(error)}"
            )
        )


# Para ejecutar esta API desde la terminal se puede utilizar:
#
# uvicorn main:app --reload
#
# main:
# Es el nombre de este archivo sin la extensión .py.
#
# app:
# Es la variable que contiene la aplicación FastAPI.
#
# --reload:
# Reinicia automáticamente el servidor cuando se modifica el código.
