"""Configuration centrale du projet.

Regroupe les chemins, les classes et les hyperparamètres afin de ne pas les
répéter dans le reste du code. Ce module n'importe ni TensorFlow ni Flask.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Arborescence
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "PlantVillage"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Fichiers produits par l'entraînement
DATASET_MANIFEST_PATH = PROCESSED_DATA_DIR / "dataset_manifest.json"
CLASS_NAMES_PATH = DATA_DIR / "class_names.json"
TRAINING_CONFIG_PATH = MODELS_DIR / "training_config.json"
MODEL_PATH = MODELS_DIR / "best_model.keras"
HISTORY_PATH = METRICS_DIR / "training_history.json"

# Livrables demandés par le cahier des charges : à la racine de results/
METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_CSV = RESULTS_DIR / "classification_report.csv"
TRAINING_CURVES_PATH = RESULTS_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

for _dossier in (DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR):
    _dossier.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes du dataset PlantVillage (38 classes, images couleur)
# ---------------------------------------------------------------------------
# L'ordre est figé : la sortie du modèle est indexée sur cette liste, et il est
# recopié dans data/class_names.json pour que l'application utilise le même.
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)

#: Libellés français des classes de tomate (les autres sont mis en forme
#: automatiquement par ``utils.helpers.nom_affichage``).
CLASSES_FR = {
    "Tomato___Early_blight": "Alternariose précoce",
    "Tomato___Late_blight": "Mildiou",
    "Tomato___Leaf_Mold": "Moisissure des feuilles",
    "Tomato___healthy": "Feuille saine",
}

# ---------------------------------------------------------------------------
# Prétraitement et entraînement
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)  # taille attendue par MobileNetV2

BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 15
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_LAYERS = 30

VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

#: Nombre minimal d'images par classe et par ensemble.
MIN_IMAGES_PER_SPLIT = 30

# ---------------------------------------------------------------------------
# Application Web
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8 Mo
MAX_IMAGE_DIMENSION = 4000

APP_NAME = "TomatoScan"
