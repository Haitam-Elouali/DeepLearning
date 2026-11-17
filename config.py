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
CONFUSION_MATRIX_NORMALIZED_PATH = RESULTS_DIR / "confusion_matrix_normalized.png"

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

#: Libellés français des 38 classes, au format « Culture — Maladie ». Ils sont
#: utilisés dans l'interface, les fiches, les figures et les rapports ; le nom
#: d'origine du jeu de données reste disponible via ``nom_anglais``.
CLASSES_FR = {
    "Apple___Apple_scab": "Pommier — Tavelure",
    "Apple___Black_rot": "Pommier — Pourriture noire",
    "Apple___Cedar_apple_rust": "Pommier — Rouille du cèdre",
    "Apple___healthy": "Pommier — Feuille saine",
    "Blueberry___healthy": "Myrtillier — Feuille saine",
    "Cherry_(including_sour)___Powdery_mildew": "Cerisier — Oïdium",
    "Cherry_(including_sour)___healthy": "Cerisier — Feuille saine",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Maïs — Cercosporiose (tache grise)",
    "Corn_(maize)___Common_rust_": "Maïs — Rouille commune",
    "Corn_(maize)___Northern_Leaf_Blight": "Maïs — Helminthosporiose",
    "Corn_(maize)___healthy": "Maïs — Feuille saine",
    "Grape___Black_rot": "Vigne — Pourriture noire",
    "Grape___Esca_(Black_Measles)": "Vigne — Esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Vigne — Brûlure des feuilles",
    "Grape___healthy": "Vigne — Feuille saine",
    "Orange___Haunglongbing_(Citrus_greening)": "Oranger — Huanglongbing",
    "Peach___Bacterial_spot": "Pêcher — Tache bactérienne",
    "Peach___healthy": "Pêcher — Feuille saine",
    "Pepper,_bell___Bacterial_spot": "Poivron — Tache bactérienne",
    "Pepper,_bell___healthy": "Poivron — Feuille saine",
    "Potato___Early_blight": "Pomme de terre — Alternariose",
    "Potato___Late_blight": "Pomme de terre — Mildiou",
    "Potato___healthy": "Pomme de terre — Feuille saine",
    "Raspberry___healthy": "Framboisier — Feuille saine",
    "Soybean___healthy": "Soja — Feuille saine",
    "Squash___Powdery_mildew": "Courge — Oïdium",
    "Strawberry___Leaf_scorch": "Fraisier — Brûlure des feuilles",
    "Strawberry___healthy": "Fraisier — Feuille saine",
    "Tomato___Bacterial_spot": "Tomate — Tache bactérienne",
    "Tomato___Early_blight": "Tomate — Alternariose précoce",
    "Tomato___Late_blight": "Tomate — Mildiou",
    "Tomato___Leaf_Mold": "Tomate — Moisissure des feuilles",
    "Tomato___Septoria_leaf_spot": "Tomate — Septoriose",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomate — Tétranyque à deux points",
    "Tomato___Target_Spot": "Tomate — Tache cible",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomate — Virus des feuilles jaunes",
    "Tomato___Tomato_mosaic_virus": "Tomate — Virus de la mosaïque",
    "Tomato___healthy": "Tomate — Feuille saine",
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

APP_NAME = "PlantDiseaseClassifier"
