"""Fonctions utilitaires : validation des fichiers, métadonnées, formatage.

Ce module ne dépend pas de TensorFlow. Il est utilisé par l'application Flask,
par les scripts d'entraînement (pour les métadonnées des classes) et par les
tests unitaires.
"""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

import config
from utils.preprocessing import charger_image_rgb

__all__ = [
    "extension_autorisee",
    "nom_fichier_securise",
    "enregistrer_upload",
    "charger_infos_maladies",
    "infos_classe",
    "nom_affichage",
    "classe_est_saine",
    "formater_date",
    "formater_pourcentage",
    "charger_json",
    "sauvegarder_json",
    "tronquer",
]


# ---------------------------------------------------------------------------
# Sécurité des fichiers téléversés
# ---------------------------------------------------------------------------
def extension_autorisee(nom_fichier: str | None) -> bool:
    """Vérifie que l'extension du fichier fait partie de la liste blanche."""
    if not nom_fichier or "." not in nom_fichier:
        return False
    extension = Path(nom_fichier).suffix.lower()
    return extension in config.ALLOWED_EXTENSIONS


def nom_fichier_securise(nom_fichier: str, longueur_max: int = 80) -> str:
    """Nettoie un nom de fichier affichable (jamais utilisé comme chemin réel).

    On supprime les caractères de contrôle et de séparation de chemin afin
    qu'un nom comme ``../../etc/passwd`` ne puisse pas être interprété.
    """
    nom = Path(str(nom_fichier or "")).name
    nom = unicodedata.normalize("NFKD", nom)
    nom = "".join(caractere for caractere in nom if not unicodedata.combining(caractere))
    nom = re.sub(r"[^A-Za-z0-9._ -]+", "_", nom)
    nom = re.sub(r"\s+", " ", nom).strip(" ._")
    if not nom:
        nom = "image"
    return nom[:longueur_max]


def enregistrer_upload(fichier: Any, dossier: Path | None = None) -> dict[str, Any]:
    """Valide puis enregistre une image téléversée.

    Sécurité appliquée :
    - extension en liste blanche (jpg, jpeg, png) ;
    - type MIME déclaré vérifié ;
    - contenu réellement décodé par Pillow (pas seulement l'extension) ;
    - dimensions maximales contrôlées ;
    - nom de fichier aléatoire (UUID) : aucune donnée du client n'est utilisée
      pour construire le chemin ;
    - ré-encodage en JPEG RGB : neutralise les métadonnées et les charges
      utiles éventuellement cachées dans le fichier d'origine.

    Returns
    -------
    dict
        ``{"nom_stocke", "nom_original", "chemin", "taille_octets"}``.
    """
    dossier = dossier or config.UPLOAD_DIR
    dossier.mkdir(parents=True, exist_ok=True)

    nom_original_brut = getattr(fichier, "filename", None) or ""
    if not nom_original_brut:
        raise ValueError("Aucun fichier n'a été sélectionné.")
    if not extension_autorisee(nom_original_brut):
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

    nom_stocke = f"{uuid.uuid4().hex}.jpg"
    chemin = dossier / nom_stocke
    image.save(chemin, format="JPEG", quality=92, optimize=True)

    return {
        "nom_stocke": nom_stocke,
        "nom_original": nom_fichier_securise(nom_original_brut),
        "chemin": chemin,
        "taille_octets": chemin.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Métadonnées des classes
# ---------------------------------------------------------------------------
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


def charger_infos_maladies() -> dict[str, dict[str, Any]]:
    """Retourne les informations pédagogiques des quatre classes."""
    return charger_json(config.DISEASE_INFO_PATH, defaut={}) or {}


def infos_classe(nom_classe: str) -> dict[str, Any]:
    """Retourne les informations d'une classe, avec repli neutre."""
    infos = charger_infos_maladies().get(nom_classe)
    if infos:
        return infos
    return {
        "nom_fr": nom_classe.replace("Tomato___", "Tomate - ").replace("_", " "),
        "nom_scientifique": "—",
        "categorie": "Non renseignée",
        "niveau": "—",
        "couleur": "#64748b",
        "resume": "Information non disponible pour cette classe.",
        "description": "",
        "symptomes": [],
        "confusion_possible": "",
        "prevention": [],
        "image_exemple": "",
    }


def nom_affichage(nom_classe: str) -> str:
    """Transforme ``Tomato___Early_blight`` en « Tomate — Alternariose précoce »."""
    return infos_classe(nom_classe).get("nom_fr") or nom_classe.replace("Tomato___", "Tomate - ")


def classe_est_saine(nom_classe: str) -> bool:
    """Indique si la classe correspond à une feuille saine."""
    return nom_classe.endswith("healthy")


# ---------------------------------------------------------------------------
# Formatage
# ---------------------------------------------------------------------------
def formater_date(valeur: str | datetime | None) -> str:
    """Formate une date ISO en « 06/11/2026 à 14:35 »."""
    if valeur is None:
        return "—"
    if isinstance(valeur, str):
        try:
            valeur = datetime.fromisoformat(valeur)
        except ValueError:
            return valeur
    return valeur.strftime("%d/%m/%Y à %H:%M")


def formater_pourcentage(valeur: float | None, decimales: int = 1) -> str:
    """Formate un score de confiance (0-1) en pourcentage lisible."""
    if valeur is None:
        return "—"
    return f"{float(valeur) * 100:.{decimales}f} %".replace(".", ",")


def tronquer(texte: str, longueur: int = 120) -> str:
    """Tronque proprement un texte trop long pour un tableau."""
    texte = str(texte or "")
    if len(texte) <= longueur:
        return texte
    return texte[: longueur - 1].rstrip() + "…"


def moyenne(valeurs: Iterable[float]) -> float:
    """Moyenne tolérante aux séquences vides."""
    valeurs = list(valeurs)
    return sum(valeurs) / len(valeurs) if valeurs else 0.0


__all__ += ["moyenne"]
