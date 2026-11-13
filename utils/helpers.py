"""Fonctions utilitaires : validation d'une image, noms des classes, formatage.

Ce module ne dépend pas de TensorFlow, ce qui permet de le tester isolément.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

import config
from utils.preprocessing import charger_image_rgb

__all__ = [
    "extension_autorisee",
    "valider_image",
    "nom_affichage",
    "classe_est_saine",
    "formater_pourcentage",
    "charger_json",
    "sauvegarder_json",
]


def extension_autorisee(nom_fichier: str | None) -> bool:
    """Vérifie que l'extension du fichier fait partie de la liste blanche."""
    if not nom_fichier or "." not in nom_fichier:
        return False
    return Path(nom_fichier).suffix.lower() in config.ALLOWED_EXTENSIONS


def valider_image(fichier: Any) -> Image.Image:
    """Valide un fichier téléversé et renvoie l'image RGB prête à l'emploi.

    Contrôles appliqués : présence du fichier, extension autorisée, type MIME
    déclaré, décodage réel par Pillow, dimensions minimales et maximales.

    Raises
    ------
    ValueError
        Avec un message affichable si l'image n'est pas exploitable.
    """
    if fichier is None or not getattr(fichier, "filename", ""):
        raise ValueError("Aucune image n'a été sélectionnée.")
    if not extension_autorisee(fichier.filename):
        raise ValueError("Format non supporté. Utilisez une image JPG, JPEG ou PNG.")

    type_mime = (getattr(fichier, "mimetype", "") or "").lower()
    if type_mime and type_mime not in config.ALLOWED_MIME_TYPES:
        raise ValueError("Type de fichier non autorisé.")

    try:
        image = charger_image_rgb(fichier.stream)
    except ValueError as exc:
        raise ValueError("Le fichier n'est pas une image lisible (fichier corrompu ?).") from exc

    largeur, hauteur = image.size
    if largeur < 32 or hauteur < 32:
        raise ValueError("Image trop petite : la largeur et la hauteur doivent dépasser 32 pixels.")
    if max(largeur, hauteur) > config.MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image trop grande : dimension maximale {config.MAX_IMAGE_DIMENSION} pixels."
        )
    return image


def nom_affichage(nom_classe: str) -> str:
    """Transforme ``Tomato___Early_blight`` en « Alternariose précoce »."""
    return config.CLASSES_FR.get(nom_classe, nom_classe.replace("Tomato___", ""))


def classe_est_saine(nom_classe: str) -> bool:
    """Indique si la classe correspond à une feuille saine."""
    return nom_classe.endswith("healthy")


def formater_pourcentage(valeur: float | None, decimales: int = 1) -> str:
    """Formate un score (0-1) en pourcentage lisible : « 95,8 % »."""
    if valeur is None:
        return "—"
    return f"{float(valeur) * 100:.{decimales}f} %".replace(".", ",")


def charger_json(chemin: Path, defaut: Any = None) -> Any:
    """Charge un fichier JSON, avec valeur par défaut si absent ou invalide."""
    try:
        with Path(chemin).open("r", encoding="utf-8") as fichier:
            return json.load(fichier)
    except (FileNotFoundError, json.JSONDecodeError):
        return defaut


def sauvegarder_json(donnees: Any, chemin: Path) -> Path:
    """Écrit un objet JSON lisible (UTF-8, indenté)."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as fichier:
        json.dump(donnees, fichier, ensure_ascii=False, indent=2)
    return chemin
