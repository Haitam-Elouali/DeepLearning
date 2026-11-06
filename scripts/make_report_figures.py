"""Génère les figures complémentaires du rapport scientifique.

Ces figures sont calculées à partir du dataset réellement téléchargé et de la
configuration d'entraînement réellement utilisée :

- dataset_examples.png      : un exemple par classe ;
- dataset_distribution.png  : répartition des classes (brute et après split) ;
- augmentation_examples.png : effet de la Data Augmentation ;
- model_architecture.png    : schéma de l'architecture du modèle.

Usage :
    python scripts/make_report_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402
from utils.helpers import charger_json, nom_affichage  # noqa: E402

plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "savefig.bbox": "tight"})
COULEURS = ["#d97706", "#b91c1c", "#7c3aed", "#15803d"]


def figure_exemples() -> Path:
    """Un exemple de feuille par classe, sur une ligne."""
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for axe, classe in zip(axes, config.CLASS_NAMES):
        fichiers = sorted(
            fichier
            for fichier in (config.RAW_DATA_DIR / classe).iterdir()
            if fichier.suffix.lower() in config.ALLOWED_EXTENSIONS
        )
        with Image.open(fichiers[0]) as image:
            axe.imshow(image.convert("RGB"))
        axe.set_title(nom_affichage(classe), fontsize=9)
        axe.axis("off")
    figure.suptitle("Exemples d'images du dataset PlantVillage (une par classe)", fontsize=11)
    figure.tight_layout()
    chemin = config.FIGURES_DIR / "dataset_examples.png"
    figure.savefig(chemin)
    plt.close(figure)
    return chemin


def figure_repartition() -> Path:
    """Répartition des images : dataset brut et ensembles préparés."""
    manifeste = charger_json(config.DATASET_MANIFEST_PATH, defaut={}) or {}
    brut = manifeste.get("images_brutes", {})
    comptages = manifeste.get("comptages", {})

    noms = [nom_affichage(classe) for classe in config.CLASS_NAMES]
    valeurs_brutes = [brut.get(classe, 0) for classe in config.CLASS_NAMES]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    barres = axes[0].bar(noms, valeurs_brutes, color=COULEURS)
    axes[0].bar_label(barres, fontsize=9)
    axes[0].set_title("Dataset brut : images par classe")
    axes[0].set_ylabel("Nombre d'images")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_ylim(0, max(valeurs_brutes) * 1.18)

    x = np.arange(len(config.CLASS_NAMES))
    largeur = 0.26
    for index, (split, etiquette, couleur) in enumerate(
        [("train", "Entraînement", "#166534"), ("validation", "Validation", "#22c55e"),
         ("test", "Test", "#0ea5e9")]
    ):
        valeurs = [comptages.get(split, {}).get(classe, 0) for classe in config.CLASS_NAMES]
        barres = axes[1].bar(x + (index - 1) * largeur, valeurs, largeur, label=etiquette,
                             color=couleur)
        axes[1].bar_label(barres, fmt="%d", fontsize=7, padding=2)
    axes[1].set_xticks(x, noms, rotation=15)
    axes[1].set_title("Après découpage stratifié (70 / 15 / 15)")
    axes[1].set_ylabel("Nombre d'images")
    axes[1].legend(ncols=3, fontsize=8)

    figure.suptitle("Répartition du dataset et des ensembles préparés", fontsize=11)
    figure.tight_layout()
    chemin = config.FIGURES_DIR / "dataset_distribution.png"
    figure.savefig(chemin)
    plt.close(figure)
    return chemin


def figure_augmentation() -> Path:
    """Illustration de la Data Augmentation sur une image réelle."""
    import keras
    from tensorflow.keras import layers

    classe = config.CLASS_NAMES[1]
    fichier = sorted(
        fichier
        for fichier in (config.RAW_DATA_DIR / classe).iterdir()
        if fichier.suffix.lower() in config.ALLOWED_EXTENSIONS
    )[0]
    with Image.open(fichier) as image:
        base = image.convert("RGB").resize(config.IMAGE_SIZE)

    augmentation = keras.Sequential(
        [
            layers.RandomRotation(factor=0.05, fill_mode="nearest"),
            layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="nearest"),
            layers.RandomFlip(mode="horizontal"),
            layers.RandomBrightness(factor=0.15, value_range=(0, 255)),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="nearest"),
        ]
    )

    tableau = np.expand_dims(np.asarray(base, dtype=np.float32), axis=0)
    figure, axes = plt.subplots(1, 5, figsize=(13, 3.2))
    axes[0].imshow(base)
    axes[0].set_title("Originale", fontsize=9)
    for index in range(1, 5):
        variante = augmentation(tableau, training=True)[0].numpy().clip(0, 255).astype("uint8")
        axes[index].imshow(variante)
        axes[index].set_title(f"Augmentée {index}", fontsize=9)
    for axe in axes:
        axe.axis("off")
    figure.suptitle(
        "Effet de la Data Augmentation (rotation, zoom, flip, luminosité, translation)",
        fontsize=10,
    )
    figure.tight_layout()
    chemin = config.FIGURES_DIR / "augmentation_examples.png"
    figure.savefig(chemin)
    plt.close(figure)
    return chemin


def figure_architecture() -> Path:
    """Schéma de l'architecture du modèle (blocs et dimensions)."""
    configuration = charger_json(config.TRAINING_CONFIG_PATH, defaut={}) or {}
    blocs = [
        ("Image d'entrée\n224 × 224 × 3", "entrée"),
        ("MobileNetV2\n(ImageNet, gelé)\n1 280 cartes 7 × 7", "base"),
        ("GlobalAverage\nPooling\n1 280", "pool"),
        ("Dropout\n20 %", "drop"),
        ("Dense ReLU\n128", "dense"),
        ("Dropout\n30 %", "drop"),
        ("Dense Softmax\n4 classes", "sortie"),
    ]

    figure, axe = plt.subplots(figsize=(13, 3.4))
    axe.set_xlim(0, 10)
    axe.set_ylim(0, 3)
    axe.axis("off")

    largeur, hauteur, espace = 1.32, 1.5, 0.06
    positions = []
    x = 0.05
    for etiquette, type_bloc in blocs:
        couleur = {
            "entrée": "#e2e8f0",
            "base": "#166534",
            "pool": "#22c55e",
            "drop": "#cbd5e1",
            "dense": "#4ade80",
            "sortie": "#b91c1c",
        }[type_bloc]
        texte = "#ffffff" if type_bloc in ("base", "sortie") else "#111827"
        boite = FancyBboxPatch(
            (x, 0.85), largeur, hauteur, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0.8, edgecolor="#94a3b8", facecolor=couleur,
        )
        axe.add_patch(boite)
        axe.text(x + largeur / 2, 0.85 + hauteur / 2, etiquette, ha="center", va="center",
                 fontsize=7.6, color=texte, fontweight="bold")
        positions.append((x, x + largeur))
        x += largeur + espace

    for (_, fin), (debut, _) in zip(positions[:-1], positions[1:]):
        axe.add_patch(
            FancyArrowPatch((fin + 0.01, 1.6), (debut - 0.01, 1.6), arrowstyle="-|>",
                            mutation_scale=9, linewidth=1, color="#475569")
        )

    parametres = configuration.get("parametres_totaux", 0)
    axe.text(
        5, 0.35,
        f"Phase 1 : base gelée — 164 484 paramètres entraînables  ·  "
        f"Phase 2 : fine-tuning des 19 dernières couches  ·  "
        f"Total : {parametres:,} paramètres".replace(",", " "),
        ha="center", fontsize=8, color="#475569",
    )
    axe.set_title("Architecture du modèle : MobileNetV2 + tête de classification", fontsize=11)
    figure.tight_layout()
    chemin = config.FIGURES_DIR / "model_architecture.png"
    figure.savefig(chemin)
    plt.close(figure)
    return chemin


def main() -> int:
    chemins = [figure_exemples(), figure_repartition(), figure_augmentation(), figure_architecture()]
    for chemin in chemins:
        print(f"Figure générée : {chemin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
