"""Génère notebooks/plant_disease_training.ipynb (notebook pédagogique).

Le notebook réutilise les modules du projet (pas de code dupliqué) afin de
rester cohérent avec les scripts d'entraînement et d'évaluation.

Usage : python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "notebooks" / "plant_disease_training.ipynb"


def markdown(texte: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texte.strip().splitlines(True)}


def code(texte: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texte.strip().splitlines(True),
    }


CELLULES = [
    markdown(
        """
# Détection des maladies de la feuille de tomate — Notebook d'expérimentation

**Projet universitaire de Deep Learning** — MobileNetV2, Transfer Learning et Flask.

Ce notebook accompagne les scripts du dossier `training/` et permet de rejouer
pas à pas la démarche expérimentale :

1. exploration du dataset PlantVillage (sous-ensemble tomate) ;
2. vérification du prétraitement et de la Data Augmentation ;
3. construction du modèle MobileNetV2 ;
4. entraînement en deux phases (extraction de caractéristiques puis fine-tuning) ;
5. évaluation sur l'ensemble de test et analyse des erreurs.

> **Avant de commencer** : télécharger le dataset
> (`python training/download_dataset.py`) puis préparer les ensembles
> (`python training/prepare_data.py`). L'entraînement complet dure environ
> 35 minutes sur un processeur à 2 cœurs ; les résultats officiels du rapport
> proviennent de `training/train.py` et `training/evaluate.py`.
"""
    ),
    code(
        """
import sys
from pathlib import Path

RACINE = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RACINE))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

print("Racine du projet :", RACINE)
print("Classes :", config.CLASS_NAMES)
print("Taille d'image :", config.IMAGE_SIZE, "| batch :", config.BATCH_SIZE)
"""
    ),
    markdown(
        """
## 1. Exploration du dataset

Les images brutes sont organisées par classe dans `data/raw/PlantVillage`.
On vérifie le nombre d'images, la résolution et la répartition des classes
(le dataset PlantVillage est déséquilibré, ce qui justifie l'usage de poids de
classes pendant l'entraînement).
"""
    ),
    code(
        """
from PIL import Image

lignes = []
for classe in config.CLASS_NAMES:
    dossier = config.RAW_DATA_DIR / classe
    fichiers = sorted(f for f in dossier.glob("*") if f.suffix.lower() in config.ALLOWED_EXTENSIONS)
    with Image.open(fichiers[0]) as exemple:
        taille = exemple.size
    lignes.append({"classe": classe, "images": len(fichiers), "resolution": f"{taille[0]}x{taille[1]}"})

df = pd.DataFrame(lignes)
df["proportion"] = (df["images"] / df["images"].sum() * 100).round(1)
df
"""
    ),
    code(
        """
figure, axe = plt.subplots(figsize=(8, 3.6))
couleurs = ["#d97706", "#b91c1c", "#7c3aed", "#15803d"]
axe.bar(df["classe"].str.replace("Tomato___", "", regex=False), df["images"], color=couleurs)
axe.set_title("Nombre d'images par classe (dataset brut)")
axe.set_ylabel("Images")
for position, valeur in enumerate(df["images"]):
    axe.text(position, valeur + 20, str(valeur), ha="center", fontsize=9)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
figure, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for axe, classe in zip(axes, config.CLASS_NAMES):
    dossier = config.RAW_DATA_DIR / classe
    fichiers = sorted(f for f in dossier.glob("*") if f.suffix.lower() in config.ALLOWED_EXTENSIONS)
    with Image.open(fichiers[0]) as image:
        axe.imshow(image.convert("RGB"))
    axe.set_title(classe.split("___")[-1].replace("_", " "), fontsize=9)
    axe.axis("off")
figure.suptitle("Un exemple par classe", fontsize=11)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## 2. Prétraitement et Data Augmentation

Le pipeline est identique à l'entraînement et à l'inférence :
**RGB → 224 × 224 → normalisation `x / 127.5 - 1`** (convention MobileNetV2).

La Data Augmentation n'est appliquée qu'à l'ensemble d'entraînement :
rotation ±5 %, zoom ±10 %, retournement horizontal, luminosité ±15 %,
translation ±5 %.
"""
    ),
    code(
        """
from utils.preprocessing import preparer_pour_modele
from training.train import construire_augmentation

classe = config.CLASS_NAMES[0]
fichier = sorted((config.PROCESSED_DATA_DIR / "train" / classe).glob("*"))[0]

tenseur = preparer_pour_modele(fichier)
print("Forme :", tenseur.shape, "| dtype :", tenseur.dtype)
print("Min / Max :", float(tenseur.min()), float(tenseur.max()))
print("Moyenne   :", float(tenseur.mean()))
"""
    ),
    code(
        """
import tensorflow as tf

with Image.open(fichier) as image:
    base = image.convert("RGB").resize(config.IMAGE_SIZE)
tableau = tf.expand_dims(np.asarray(base, dtype=np.float32), 0)

augmentation = construire_augmentation()
figure, axes = plt.subplots(1, 5, figsize=(13, 3.2))
axes[0].imshow(base)
axes[0].set_title("Originale", fontsize=9)
for index in range(1, 5):
    variante = augmentation(tableau, training=True)[0].numpy().clip(0, 255).astype("uint8")
    axes[index].imshow(variante)
    axes[index].set_title(f"Augmentée {index}", fontsize=9)
for axe in axes:
    axe.axis("off")
figure.suptitle("Effet de la Data Augmentation", fontsize=11)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## 3. Construction du modèle

Architecture : MobileNetV2 (ImageNet, sans sa couche de classification)
→ GlobalAveragePooling2D → Dropout(0.2) → Dense(128, ReLU) → Dropout(0.3)
→ Dense(4, Softmax).
"""
    ),
    code(
        """
from training.train import construire_modele, poids_des_classes

modele, base = construire_modele(config.NUM_CLASSES, learning_rate=config.LEARNING_RATE)
modele.summary()

print("\\nPoids de classes (compensation du déséquilibre) :")
print(poids_des_classes(config.PROCESSED_DATA_DIR / "train"))
"""
    ),
    markdown(
        """
## 4. Entraînement

Deux phases sont prévues par `training/train.py` :

- **phase 1** : base gelée, Adam (1e-3), Early Stopping et Model Checkpoint ;
- **phase 2** : fine-tuning des 30 dernières couches, Adam (1e-5).

L'entraînement complet est long sur CPU. Pour le rejouer ici, décommenter la
cellule suivante (environ 35 minutes). Les résultats officiels se trouvent dans
`results/metrics/evaluation.json` et `results/figures/`.
"""
    ),
    code(
        """
# import subprocess
# subprocess.run([sys.executable, "training/train.py", "--epochs", "15", "--fine-tune-epochs", "8"], cwd=RACINE)
"""
    ),
    markdown(
        """
## 5. Courbes d'apprentissage

Les courbes sont produites par `training/evaluate.py` à partir de l'historique
réel sauvegardé dans `results/metrics/training_history.json`.
"""
    ),
    code(
        """
figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
if config.HISTORY_PATH.exists():
    from utils.helpers import charger_json
    historique = charger_json(config.HISTORY_PATH)
    epochs = range(1, len(historique["loss"]) + 1)
    axes[0].plot(epochs, historique["accuracy"], label="Entraînement", color="#15803d")
    axes[0].plot(epochs, historique["val_accuracy"], label="Validation", color="#b91c1c")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Époque"); axes[0].legend()
    axes[1].plot(epochs, historique["loss"], label="Entraînement", color="#15803d")
    axes[1].plot(epochs, historique["val_loss"], label="Validation", color="#b91c1c")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Époque"); axes[1].legend()
else:
    print("Historique absent : lancez training/train.py")
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## 6. Évaluation sur l'ensemble de test

Les métriques sont calculées à partir des prédictions réelles du modèle sur les
images de `data/processed/test` (jamais vues à l'entraînement).
"""
    ),
    code(
        """
from utils.helpers import charger_json

if (config.METRICS_DIR / "evaluation.json").exists():
    metriques = charger_json(config.METRICS_DIR / "evaluation.json")
    pd.DataFrame([
        {"métrique": "Accuracy", "valeur": round(metriques["global"]["accuracy"], 4)},
        {"métrique": "Precision (macro)", "valeur": round(metriques["global"]["precision_macro"], 4)},
        {"métrique": "Recall (macro)", "valeur": round(metriques["global"]["recall_macro"], 4)},
        {"métrique": "F1-score (macro)", "valeur": round(metriques["global"]["f1_macro"], 4)},
    ])
else:
    print("Lancez training/evaluate.py pour générer les métriques.")
"""
    ),
    code(
        """
if (config.METRICS_DIR / "evaluation.json").exists():
    pd.DataFrame(metriques["par_classe"]).T
"""
    ),
    code(
        """
from PIL import Image

chemins = [config.FIGURES_DIR / "confusion_matrix.png", config.FIGURES_DIR / "per_class_performance.png"]
for chemin in chemins:
    if chemin.exists():
        with Image.open(chemin) as image:
            plt.figure(figsize=(12, 5))
            plt.imshow(image)
            plt.axis("off")
            plt.show()
    else:
        print("Figure absente :", chemin.name)
"""
    ),
    markdown(
        """
## 7. Prédiction sur une nouvelle image

Le prédicteur réutilise **exactement** le même prétraitement que l'entraînement
(`utils/preprocessing.py`), ce qui évite tout décalage de distribution.
"""
    ),
    code(
        """
from utils.predictor import obtenir_predictor
from utils.helpers import nom_affichage

predictor = obtenir_predictor()
predictor.charger()

resultat = predictor.predire(fichier)
print("Classe prédite :", nom_affichage(resultat["classe"]))
print(f"Confiance : {resultat['confiance'] * 100:.1f} %")
print(f"Durée : {resultat['duree_ms']} ms")
for element in resultat["classement"]:
    print(f"  {nom_affichage(element['classe']):<28} {element['probabilite'] * 100:5.1f} %")
"""
    ),
    markdown(
        """
## 8. Limites et perspectives

- Quatre classes seulement ; les autres maladies de la tomate sont hors périmètre.
- Images de laboratoire (fond uniforme) : la généralisation à des photos de terrain
  n'est pas garantie.
- Déséquilibre entre classes et similarité visuelle entre mildiou et alternariose.
- Perspectives : ajout de classes, validation croisée, Grad-CAM pour visualiser les
  zones discriminantes, test sur photos de terrain, quantification INT8 pour un
  déploiement mobile.
"""
    ),
]

NOTEBOOK = {
    "cells": CELLULES,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> int:
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook écrit : {SORTIE} ({len(CELLULES)} cellules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
