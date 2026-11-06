"""Configuration centrale du projet de détection des maladies de la tomate.

Tous les chemins et hyperparamètres du projet sont regroupés ici afin de
pouvoir adapter le comportement des scripts d'entraînement, d'évaluation et
de l'application Web sans modifier le reste du code.

Le module ne dépend d'aucune bibliothèque lourde (ni TensorFlow, ni Flask) :
il peut donc être importé partout, y compris par les tests.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Arborescence du projet
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

STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
EXAMPLES_DIR = STATIC_DIR / "images" / "examples"

# Fichiers de données produits par l'entraînement
CLASS_NAMES_PATH = DATA_DIR / "class_names.json"
DATASET_MANIFEST_PATH = DATA_DIR / "processed" / "dataset_manifest.json"
TRAINING_CONFIG_PATH = MODELS_DIR / "training_config.json"
MODEL_PATH = MODELS_DIR / "best_model.keras"
HISTORY_PATH = METRICS_DIR / "training_history.json"

# Métadonnées descriptives (rédigées à la main, sans dépendance au dataset)
DISEASE_INFO_PATH = DATA_DIR / "disease_info.json"

# Base de données de l'historique des analyses
DATABASE_PATH = Path(os.environ.get("PLANT_DB_PATH", DATA_DIR / "history.db"))

for _directory in (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    UPLOAD_DIR,
    EXAMPLES_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes du problème
# ---------------------------------------------------------------------------
# Le nom des dossiers est celui du dataset PlantVillage d'origine. L'ordre
# alphabétique est celui utilisé par `image_dataset_from_directory`, il est
# sauvegardé dans data/class_names.json pour garantir la cohérence entre
# l'entraînement et l'inférence.
CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)

#: Nombre d'images attendu par classe dans le dataset PlantVillage d'origine.
#: Utilisé uniquement pour les contrôles de cohérence du script de
#: téléchargement (le dataset complet n'est pas versionné).
EXPECTED_IMAGES_PER_CLASS = {
    "Tomato___Early_blight": 1000,
    "Tomato___Late_blight": 1909,
    "Tomato___Leaf_Mold": 952,
    "Tomato___healthy": 1591,
}

# ---------------------------------------------------------------------------
# Prétraitement et entraînement
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)  # taille attendue par MobileNetV2
IMAGE_HEIGHT, IMAGE_WIDTH = IMAGE_SIZE

BATCH_SIZE = 32
EPOCHS = 30  # nombre maximal d'époques pour la phase 1 (Transfer Learning)
FINE_TUNE_EPOCHS = 15  # nombre maximal d'époques pour la phase 2 (Fine-Tuning)
LEARNING_RATE = 1e-3  # phase 1 : seules les couches ajoutées apprennent
FINE_TUNE_LEARNING_RATE = 1e-5  # phase 2 : apprentissage très lent
FINE_TUNE_LAYERS = 30  # nombre de couches dégelées en fin de MobileNetV2

VALIDATION_SPLIT = 0.15  # 15 % validation
TEST_SPLIT = 0.15  # 15 % test (indépendant, jamais vu à l'entraînement)
SEED = 42  # graine globale pour la reproductibilité

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

#: Taille maximale du sous-échantillon par classe pour l'entraînement.
#: ``None`` = dataset complet. Ce paramètre existe pour permettre un
#: entraînement rapide sur une machine modeste (CPU) tout en documentant
#: précisément le sous-ensemble utilisé dans le rapport.
MAX_IMAGES_PER_CLASS: int | None = None
#: Nombre d'images conservées au minimum par classe dans chaque ensemble.
MIN_IMAGES_PER_SPLIT = 30

# ---------------------------------------------------------------------------
# Application Web
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8 Mo
MAX_IMAGE_DIMENSION = 4000  # refus des images démesurées (sécurité)
HISTORY_PAGE_SIZE = 50

#: Clé de session Flask : surchargeable par variable d'environnement.
SECRET_KEY = os.environ.get("PLANT_SECRET_KEY", "dev-secret-a-changer-en-production")

#: Version affichée dans l'interface et stockée avec chaque analyse.
APP_NAME = "TomatoScan"
APP_FULL_NAME = "Détection des maladies de la feuille de tomate"
APP_VERSION = "1.0.0"
