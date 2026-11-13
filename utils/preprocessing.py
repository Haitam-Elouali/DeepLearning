"""Prétraitement des images partagé par l'entraînement et l'inférence.

Règle d'or du projet : **le prétraitement appliqué à la prédiction doit être
rigoureusement identique à celui appliqué pendant l'entraînement**. Toutes les
opérations sont donc centralisées dans ce module, qui ne dépend que de Pillow
et de NumPy (pas de TensorFlow), afin de pouvoir être testé isolément.

Pipeline appliqué :
1. ouverture de l'image et conversion en RGB ;
2. redimensionnement à 224 x 224 pixels ;
3. conversion en tableau ``float32`` dont les valeurs restent dans [0, 255] ;
4. normalisation ``x / 127.5 - 1`` → intervalle [-1, 1].

La dernière étape reproduit exactement
``tensorflow.keras.applications.mobilenet_v2.preprocess_input`` pour une entrée
en nombres flottants ; le test ``tests/test_preprocessing.py`` vérifie cette
équivalence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import IMAGE_SIZE

__all__ = [
    "IMAGE_SIZE",
    "charger_image_rgb",
    "redimensionner",
    "image_vers_tableau",
    "normaliser",
    "preparer_pour_modele",
    "preparer_image",
    "preparer_lot",
]


def charger_image_rgb(source: str | Path | object) -> Image.Image:
    """Ouvre une image et la convertit en RGB.

    Parameters
    ----------
    source:
        Chemin vers le fichier ou objet fichier (``FileStorage`` de Flask).

    Raises
    ------
    ValueError
        Si le fichier n'est pas une image exploitable.
    """
    try:
        image = Image.open(source)
        # Appeler load() force la lecture réelle des données : un fichier
        # corrompu ou une fausse extension est ainsi détecté immédiatement.
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Le fichier fourni n'est pas une image valide.") from exc

    # EXIF : applique l'orientation du téléphone avant tout traitement.
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def redimensionner(image: Image.Image, taille: tuple[int, int] = IMAGE_SIZE) -> Image.Image:
    """Redimensionne l'image à la taille attendue par le modèle."""
    if image.size == taille:
        return image
    return image.resize(taille, Image.Resampling.BILINEAR)


def image_vers_tableau(image: Image.Image) -> np.ndarray:
    """Convertit une image RGB en tableau ``float32`` entre 0 et 255."""
    return np.asarray(image, dtype=np.float32)


def normaliser(tableau: np.ndarray) -> np.ndarray:
    """Normalise un tableau [0, 255] vers [-1, 1] (convention MobileNetV2)."""
    return (np.asarray(tableau, dtype=np.float32) / 127.5) - 1.0


def preparer_pour_modele(
    source: str | Path | object,
    taille: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Applique le pipeline complet et renvoie une image prête pour le modèle."""
    image = charger_image_rgb(source)
    image = redimensionner(image, taille)
    return normaliser(image_vers_tableau(image))


def preparer_image(
    image: Image.Image,
    taille: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Prépare une image déjà ouverte (pas de relecture du fichier)."""
    return normaliser(image_vers_tableau(redimensionner(image, taille)))


def preparer_lot(
    sources: Iterable[str | Path | object],
    taille: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Prépare un lot d'images (utile pour l'évaluation et l'analyse d'erreurs)."""
    tenseurs: Sequence[np.ndarray] = [preparer_pour_modele(s, taille) for s in sources]
    return np.stack(tenseurs, axis=0)
