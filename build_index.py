# Importa json para leer y crear archivos en formato JSON.
import json

# Importa Path para trabajar con rutas de archivos
# de forma compatible con diferentes sistemas operativos.
from pathlib import Path

# Importa NumPy para almacenar y manipular los vectores
# numéricos generados por el modelo.
import numpy as np

# Importa PyTorch, utilizado por el modelo CLIP
# para realizar operaciones de inteligencia artificial.
import torch

# Importa Image de Pillow para abrir y convertir imágenes.
from PIL import Image

# Importa el procesador y el modelo CLIP desde Transformers.
#
# CLIPProcessor prepara las imágenes antes de enviarlas al modelo.
# CLIPModel genera una representación numérica de cada imagen.
from transformers import CLIPProcessor, CLIPModel


# Obtiene la ruta absoluta de la carpeta donde se encuentra
# este archivo de Python.
BASE_DIR = Path(__file__).resolve().parent

# Define la ubicación del archivo que contiene
# los personajes y las rutas de sus imágenes.
METADATA_PATH = BASE_DIR / "metadata.json"

# Define la ubicación donde se guardarán
# los vectores numéricos de las imágenes.
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"

# Define la ubicación donde se guardarán las etiquetas
# relacionadas con cada vector generado.
LABELS_PATH = BASE_DIR / "labels.json"

# Nombre del modelo CLIP preentrenado que se descargará
# desde Hugging Face.
MODEL_NAME = "openai/clip-vit-base-patch32"


def load_model():
    """
    Carga el modelo CLIP y su procesador.

    El modelo utilizará la tarjeta gráfica mediante CUDA
    si está disponible. En caso contrario, utilizará el procesador.

    Retorna:
        model: Modelo CLIP cargado.
        processor: Procesador encargado de preparar las imágenes.
        device: Dispositivo utilizado, ya sea "cuda" o "cpu".
    """

    # Comprueba si PyTorch puede utilizar una tarjeta gráfica
    # compatible con CUDA.
    #
    # Si CUDA está disponible, se utiliza la GPU.
    # En caso contrario, se utiliza el procesador.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Muestra el dispositivo seleccionado.
    print(f"Usando dispositivo: {device}")

    # Informa que comenzará la carga del modelo.
    print("Cargando modelo CLIP...")

    # Descarga o carga desde la memoria caché
    # el modelo CLIP preentrenado.
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Descarga o carga el procesador correspondiente al modelo.
    #
    # Este procesador redimensiona, normaliza y convierte
    # las imágenes en tensores.
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Mueve el modelo al dispositivo seleccionado.
    #
    # Puede ser la GPU mediante CUDA o el procesador mediante CPU.
    model.to(device)

    # Coloca el modelo en modo evaluación.
    #
    # Esto desactiva comportamientos utilizados únicamente
    # durante el entrenamiento, como algunas capas aleatorias.
    model.eval()

    # Retorna los elementos necesarios para procesar imágenes.
    return model, processor, device


def encode_image(image_path: Path, model, processor, device):
    """
    Convierte una imagen en un vector numérico llamado embedding.

    Parámetros:
        image_path (Path):
            Ruta de la imagen que se desea procesar.

        model:
            Modelo CLIP utilizado para generar el embedding.

        processor:
            Procesador que prepara la imagen antes de enviarla al modelo.

        device:
            Dispositivo donde se ejecutará el modelo: CPU o CUDA.

    Retorna:
        numpy.ndarray:
            Vector de tipo float32 que representa las características
            visuales de la imagen.
    """

    # Abre la imagen y la convierte al formato RGB.
    #
    # Esto evita problemas con imágenes en escala de grises,
    # con transparencia o con otros formatos de color.
    image = Image.open(image_path).convert("RGB")

    # Prepara la imagen para ser procesada por CLIP.
    inputs = processor(
        images=image,

        # Indica que el resultado debe convertirse
        # en tensores de PyTorch.
        return_tensors="pt"
    )

    # Obtiene los valores numéricos de los píxeles
    # preparados por el procesador.
    #
    # Después, los mueve al dispositivo seleccionado.
    pixel_values = inputs["pixel_values"].to(device)

    # Desactiva el cálculo de gradientes.
    #
    # Los gradientes solamente son necesarios al entrenar un modelo.
    # Como aquí solo se utilizan imágenes para obtener embeddings,
    # desactivarlos reduce el consumo de memoria y mejora el rendimiento.
    with torch.no_grad():

        # Envía los píxeles al modelo CLIP para generar
        # las características visuales de la imagen.
        features = model.get_image_features(
            pixel_values=pixel_values
        )

        # Algunas versiones de Transformers pueden devolver
        # un objeto en lugar de un tensor directamente.
        #
        # Por eso se comprueba el tipo de resultado.
        if not isinstance(features, torch.Tensor):

            # Intenta obtener el embedding desde image_embeds.
            if (
                hasattr(features, "image_embeds")
                and features.image_embeds is not None
            ):
                features = features.image_embeds

            # Si image_embeds no existe, intenta utilizar pooler_output.
            elif (
                hasattr(features, "pooler_output")
                and features.pooler_output is not None
            ):
                features = features.pooler_output

            # Si no se encuentra ningún resultado válido,
            # se genera una excepción.
            else:
                raise RuntimeError(
                    "No se pudo obtener el embedding de la imagen. "
                    f"Tipo recibido: {type(features)}"
                )

    # Normaliza el vector mediante la norma L2.
    #
    # La normalización hace que el vector tenga una longitud igual a 1.
    # Esto permite comparar embeddings mediante similitud coseno
    # de una manera más consistente.
    features = features / features.norm(
        p=2,
        dim=-1,
        keepdim=True
    )

    # Mueve el tensor a la CPU, lo convierte en un arreglo de NumPy,
    # toma el primer elemento del lote y lo convierte a float32.
    #
    # El procesador crea un lote aunque se procese una sola imagen,
    # por eso se utiliza el índice [0].
    return features.cpu().numpy()[0].astype("float32")


def main():
    """
    Ejecuta el proceso completo de creación del índice de imágenes.

    El proceso realiza los siguientes pasos:

    1. Verifica que exista metadata.json.
    2. Lee los personajes y las imágenes.
    3. Carga el modelo CLIP.
    4. Procesa cada imagen.
    5. Genera un embedding por cada imagen.
    6. Guarda los embeddings en embeddings.npy.
    7. Guarda las etiquetas en labels.json.
    """

    # Verifica que exista el archivo metadata.json.
    if not METADATA_PATH.exists():

        # Detiene el programa si el archivo no existe.
        raise FileNotFoundError(
            "No existe metadata.json"
        )

    # Abre metadata.json en modo lectura.
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        # Convierte el contenido JSON en una estructura de Python.
        metadata = json.load(file)

    # Carga el modelo CLIP, el procesador y el dispositivo.
    model, processor, device = load_model()

    # Lista donde se guardarán los vectores numéricos
    # generados para cada imagen.
    embeddings = []

    # Lista donde se almacenará la información relacionada
    # con cada embedding.
    labels = []

    # Recorre cada personaje registrado en metadata.json.
    for item in metadata:

        # Obtiene el nombre del personaje.
        character_name = item.get("character")

        # Obtiene la lista de imágenes del personaje.
        #
        # Si no existe la propiedad "images",
        # se utiliza una lista vacía.
        images = item.get("images", [])

        # Verifica que el registro tenga un nombre de personaje.
        if not character_name:

            # Muestra una advertencia y continúa
            # con el siguiente registro.
            print(
                "Registro sin nombre de personaje. Se omite."
            )
            continue

        # Recorre todas las imágenes asociadas al personaje.
        for relative_image_path in images:

            # Construye la ruta absoluta de la imagen.
            image_path = BASE_DIR / relative_image_path

            # Verifica que la imagen exista.
            if not image_path.exists():

                # Muestra una advertencia y continúa
                # con la siguiente imagen.
                print(
                    f"No existe la imagen: {image_path}"
                )
                continue

            # Informa qué personaje e imagen se están procesando.
            print(
                f"Procesando: {character_name} "
                f"-> {relative_image_path}"
            )

            # Convierte la imagen en un embedding utilizando CLIP.
            embedding = encode_image(
                image_path,
                model,
                processor,
                device
            )

            # Agrega el embedding generado a la lista.
            embeddings.append(embedding)

            # Agrega la etiqueta correspondiente al embedding.
            #
            # El orden de esta lista debe coincidir con el orden
            # de los vectores almacenados en embeddings.
            labels.append({
                "character": character_name,
                "sourceImage": relative_image_path
            })

    # Verifica que al menos una imagen haya sido procesada.
    if len(embeddings) == 0:

        # Detiene el programa si no se generó ningún embedding.
        raise ValueError(
            "No se generó ningún embedding. "
            "Revisa metadata.json e imágenes."
        )

    # Combina todos los vectores individuales en una matriz.
    #
    # Cada fila de la matriz representa una imagen diferente.
    embeddings_array = np.vstack(embeddings)

    # Guarda la matriz de embeddings en un archivo binario de NumPy.
    #
    # Este archivo puede cargarse posteriormente con np.load().
    np.save(
        EMBEDDINGS_PATH,
        embeddings_array
    )

    # Abre o crea labels.json en modo escritura.
    with open(
        LABELS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        # Guarda las etiquetas en formato JSON.
        #
        # ensure_ascii=False permite conservar caracteres especiales.
        # indent=2 organiza el contenido con dos espacios de indentación.
        json.dump(
            labels,
            file,
            ensure_ascii=False,
            indent=2
        )

    # Informa que el índice terminó de generarse.
    print("Índice generado correctamente.")

    # Muestra la cantidad total de imágenes procesadas.
    print(
        f"Total de imágenes indexadas: {len(labels)}"
    )

    # Muestra la ubicación del archivo de embeddings.
    print(
        f"Archivo generado: {EMBEDDINGS_PATH}"
    )

    # Muestra la ubicación del archivo de etiquetas.
    print(
        f"Archivo generado: {LABELS_PATH}"
    )


# Comprueba si este archivo se está ejecutando directamente.
#
# Evita que main() se ejecute automáticamente
# cuando este archivo sea importado desde otro módulo.
if __name__ == "__main__":

    # Inicia la creación del índice de imágenes.
    main()

