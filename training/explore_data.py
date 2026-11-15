"""Analyse exploratoire du dataset : figures du rapport.

Produit quatre figures à partir des données réellement présentes dans
``data/raw/PlantVillage`` et ``data/processed`` :

- ``dataset_examples.png``     : une image par classe ;
- ``dataset_distribution.png`` : effectifs par classe, avant et après découpage ;
- ``augmentation_examples.png``: effet des transformations d'augmentation ;
- ``model_architecture.png``   : schéma de l'architecture du modèle.

Usage :
    python training/explore_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402
from utils.helpers import charger_json, nom_affichage  # noqa: E402

VERT_FONCE = "#14532d"
VERT = "#15803d"
VERT_CLAIR = "#22c55e"
ROUGE = "#b91c1c"
GRIS = "#4b5563"
GRIS_CLAIR = "#e5e7eb"


def _premiere_image(classe: str) -> Path | None:
    """Renvoie la première image trouvée pour une classe, ou None."""
    dossier = config.RAW_DATA_DIR / classe
    if not dossier.is_dir():
        return None
    for chemin in sorted(dossier.iterdir()):
        if chemin.suffix.lower() in config.ALLOWED_EXTENSIONS:
            return chemin
    return None


def image_exemples() -> None:
    """Grille : une image représentative par classe."""
    images = [(classe, _premiere_image(classe)) for classe in config.CLASS_NAMES]
    images = [(classe, chemin) for classe, chemin in images if chemin is not None]
    if not images:
        raise SystemExit(f"Aucune image trouvée dans {config.RAW_DATA_DIR}")

    colonnes = 6
    lignes = int(np.ceil(len(images) / colonnes))
    figure, axes = plt.subplots(lignes, colonnes, figsize=(colonnes * 2.3, lignes * 2.5))
    axes = np.atleast_1d(axes).ravel()

    for axe, (classe, chemin) in zip(axes, images):
        axe.imshow(Image.open(chemin).convert("RGB"))
        axe.set_title(nom_affichage(classe), fontsize=7)
        axe.axis("off")
    for axe in axes[len(images):]:
        axe.axis("off")

    figure.suptitle(
        f"Une image par classe — {len(images)} classes du dataset PlantVillage",
        fontsize=13,
        color=VERT_FONCE,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(config.FIGURES_DIR / "dataset_examples.png", dpi=100)
    plt.close(figure)


def distribution() -> None:
    """Diagramme en barres des effectifs, empilés par ensemble."""
    manifeste = charger_json(config.DATASET_MANIFEST_PATH, defaut={}) or {}
    comptages = manifeste.get("comptages", {})
    if not comptages:
        raise SystemExit("dataset_manifest.json absent : lancez training/prepare_data.py")

    classes = sorted(config.CLASS_NAMES, key=lambda c: manifeste["images_brutes"].get(c, 0))
    train = [comptages["train"].get(c, 0) for c in classes]
    validation = [comptages["validation"].get(c, 0) for c in classes]
    test = [comptages["test"].get(c, 0) for c in classes]

    figure, axe = plt.subplots(figsize=(11, 11))
    positions = np.arange(len(classes))
    axe.barh(positions, train, color=VERT, label=f"Entraînement ({sum(train)})")
    axe.barh(
        positions,
        validation,
        left=train,
        color=VERT_CLAIR,
        label=f"Validation ({sum(validation)})",
    )
    axe.barh(
        positions,
        test,
        left=np.array(train) + np.array(validation),
        color="#86efac",
        label=f"Test ({sum(test)})",
    )
    axe.set_yticks(positions)
    axe.set_yticklabels([nom_affichage(c) for c in classes], fontsize=7)
    axe.set_xlabel("Nombre d'images")
    axe.set_title(
        f"Effectifs par classe après découpage 70 / 15 / 15 — {sum(train) + sum(validation) + sum(test)} images",
        fontsize=12,
        color=VERT_FONCE,
    )
    axe.legend(loc="lower right", fontsize=9)
    axe.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(config.FIGURES_DIR / "dataset_distribution.png", dpi=130)
    plt.close(figure)


def augmentation() -> None:
    """Une image et ses variantes augmentées."""
    import tensorflow as tf

    classe = "Tomato___Late_blight"
    chemin = _premiere_image(classe) or _premiere_image(config.CLASS_NAMES[0])
    if chemin is None:
        raise SystemExit("Aucune image disponible pour illustrer l'augmentation")

    image = tf.keras.utils.load_img(chemin, target_size=config.IMAGE_SIZE)
    tableau = tf.keras.utils.img_to_array(image)[None, ...]

    couches = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05, seed=config.SEED),
            tf.keras.layers.RandomZoom(0.1, seed=config.SEED),
            tf.keras.layers.RandomFlip("horizontal", seed=config.SEED),
            tf.keras.layers.RandomBrightness(0.15, seed=config.SEED),
            tf.keras.layers.RandomTranslation(0.05, 0.05, seed=config.SEED),
        ]
    )

    figure, axes = plt.subplots(1, 6, figsize=(15, 3.1))
    axes[0].imshow(tableau[0].astype("uint8"))
    axes[0].set_title("Originale", fontsize=10)
    axes[0].axis("off")
    for index in range(1, 6):
        variante = couches(tableau, training=True)[0].numpy()
        variante = np.clip(variante, 0, 255).astype("uint8")
        axes[index].imshow(variante)
        axes[index].set_title(f"Variante {index}", fontsize=10)
        axes[index].axis("off")

    figure.suptitle(
        f"Data Augmentation — {nom_affichage(classe)} (appliquée uniquement à l'entraînement)",
        fontsize=12,
        color=VERT_FONCE,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(config.FIGURES_DIR / "augmentation_examples.png", dpi=130)
    plt.close(figure)


def architecture() -> None:
    """Schéma des blocs du modèle, avec le nombre réel de classes."""
    configuration = charger_json(config.TRAINING_CONFIG_PATH, defaut={}) or {}
    total = configuration.get("parametres_totaux", 0)
    entrainables = configuration.get("parametres_entrainables_phase1", 0)

    blocs = [
        (f"Image d'entrée\n{config.IMAGE_SIZE[0]} × {config.IMAGE_SIZE[1]} × 3", GRIS_CLAIR, GRIS),
        ("MobileNetV2\n(ImageNet)\n1 280 cartes 7 × 7", VERT_FONCE, "white"),
        ("GlobalAverage\nPooling\n1 280", VERT_CLAIR, "black"),
        ("Dropout\n20 %", GRIS_CLAIR, GRIS),
        ("Dense ReLU\n128", VERT_CLAIR, "black"),
        ("Dropout\n30 %", GRIS_CLAIR, GRIS),
        (f"Dense Softmax\n{config.NUM_CLASSES} classes", ROUGE, "white"),
    ]

    figure, axe = plt.subplots(figsize=(14, 4.2))
    largeur, hauteur, ecart = 1.9, 1.5, 0.32
    for index, (texte, couleur, texte_couleur) in enumerate(blocs):
        gauche = index * (largeur + ecart)
        boite = FancyBboxPatch(
            (gauche, 0.6),
            largeur,
            hauteur,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=couleur,
            edgecolor="#94a3b8" if couleur == GRIS_CLAIR else couleur,
            linewidth=1.2,
        )
        axe.add_patch(boite)
        axe.text(
            gauche + largeur / 2,
            0.6 + hauteur / 2,
            texte,
            ha="center",
            va="center",
            fontsize=9,
            color=texte_couleur,
            fontweight="bold" if couleur != GRIS_CLAIR else "normal",
        )
        if index < len(blocs) - 1:
            axe.add_patch(
                FancyArrowPatch(
                    (gauche + largeur, 0.6 + hauteur / 2),
                    (gauche + largeur + ecart, 0.6 + hauteur / 2),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    color=GRIS,
                    linewidth=1.1,
                )
            )

    axe.set_xlim(-0.15, len(blocs) * (largeur + ecart) - ecart + 0.15)
    axe.set_ylim(0.25, 2.75)
    axe.axis("off")
    axe.set_title(
        f"Architecture du modèle : MobileNetV2 + tête de classification "
        f"({config.NUM_CLASSES} classes)",
        fontsize=13,
        color=VERT_FONCE,
    )
    axe.text(
        len(blocs) * (largeur + ecart) / 2 - ecart / 2,
        0.35,
        f"Phase 1 : base gelée — {entrainables:,} paramètres entraînables · "
        f"Phase 2 : fine-tuning des dernières couches · Total : {total:,} paramètres".replace(",", " "),
        ha="center",
        fontsize=9,
        color=GRIS,
    )
    figure.tight_layout()
    figure.savefig(config.FIGURES_DIR / "model_architecture.png", dpi=130)
    plt.close(figure)


def main() -> int:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    image_exemples()
    distribution()
    augmentation()
    architecture()
    print(f"Figures d'exploration générées dans {config.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
