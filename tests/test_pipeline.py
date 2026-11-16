"""Tests du pipeline : prétraitement entraînement == inférence.

Le point critique est l'absence de double normalisation et l'identité stricte
entre le pipeline ``tf.data`` de l'entraînement et celui de l'application.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import config

keras = pytest.importorskip("tensorflow.keras")
tf = pytest.importorskip("tensorflow")

from training.train import (  # noqa: E402
    charger_ensembles,
    construire_augmentation,
    decrire_architecture,
    resume_fine_tuning,
)
from utils.preprocessing import preparer_pour_modele  # noqa: E402


def _image_motif(chemin: Path, indice: int, numero: int) -> Path:
    """Écrit une image synthétique à dégradé et motifs (pas de couleur unie).

    La taille (300 x 300) diffère volontairement de 224 x 224 : le
    redimensionnement est ainsi réellement exercé, comme sur PlantVillage
    (256 x 256 -> 224 x 224).
    """
    from PIL import Image, ImageDraw

    taille = 300
    yy, xx = np.mgrid[0:taille, 0:taille]
    tableau = np.stack(
        [
            (xx * 2 + indice * 37) % 256,
            (yy * 3 + numero * 53) % 256,
            (xx + yy) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)

    image = Image.fromarray(tableau, mode="RGB")
    trace = ImageDraw.Draw(image)
    trace.rectangle([20, 20, 90, 90], fill=(255, 255, 255))
    trace.line([0, taille - 1, taille - 1, 0], fill=(0, 0, 0), width=3)
    trace.ellipse([120, 120, 250, 250], outline=(200, 30, 30), width=3)

    chemin.parent.mkdir(parents=True, exist_ok=True)
    image.save(chemin, format="PNG")
    return chemin


def _jeu_reduit(dossier: Path, par_classe: int = 2) -> dict[str, list[Path]]:
    """Crée un mini-dataset (train/validation/test) avec les 38 classes."""
    images: dict[str, list[Path]] = {}
    for indice, classe in enumerate(config.CLASS_NAMES):
        fichiers = []
        for split in ("train", "validation", "test"):
            dossier_classe = dossier / split / classe
            for numero in range(par_classe):
                fichiers.append(
                    _image_motif(dossier_classe / f"{numero}.png", indice, numero)
                )
        images[classe] = fichiers
    return images


@pytest.fixture(scope="module")
def mini_dataset(tmp_path_factory):
    dossier = tmp_path_factory.mktemp("mini")
    images = _jeu_reduit(dossier)
    return dossier, images


def test_validation_et_test_utilisent_exactement_le_preprocessing_d_inference(
    mini_dataset,
):
    """Chaque image du pipeline doit être *le* tenseur de l'inférence.

    La comparaison est faite image par image, dans l'ordre des dossiers
    (``shuffle=False``), et non en cherchant la meilleure correspondance parmi
    toutes les images : une erreur d'ordre ou de redimensionnement est donc
    détectée.
    """
    dossier, _ = mini_dataset
    _, val_ds, test_ds = charger_ensembles(dossier, batch_size=8, graine=42)

    for dataset, split in ((val_ds, "validation"), (test_ds, "test")):
        attendus = [
            preparer_pour_modele(chemin)
            for classe in config.CLASS_NAMES
            for chemin in sorted((dossier / split / classe).iterdir())
        ]
        obtenus = [image for lot, _ in dataset for image in lot.numpy()]

        assert len(obtenus) == len(attendus), split
        for index, (obtenu, attendu) in enumerate(zip(obtenus, attendus)):
            assert obtenu.shape == (*config.IMAGE_SIZE, 3)
            ecart = float(np.max(np.abs(obtenu - attendu)))
            assert ecart < 1e-5, f"{split}, image {index} : écart {ecart}"


def test_un_seul_algorithme_de_redimensionnement(mini_dataset):
    """Le fichier et l'image déjà ouverte passent par le même redimensionnement.

    Le pipeline ``tf.data`` appelle ``preparer_pour_modele`` : il n'y a donc
    qu'une seule implémentation du prétraitement dans tout le projet (Pillow),
    partagée par l'entraînement, la validation, le test et l'inférence.
    """
    from PIL import Image

    from utils.preprocessing import preparer_image

    dossier, _ = mini_dataset
    chemin = sorted((dossier / "validation" / config.CLASS_NAMES[0]).iterdir())[0]

    depuis_fichier = preparer_pour_modele(chemin)
    with Image.open(chemin) as image:
        depuis_image = preparer_image(image.convert("RGB"))

    assert depuis_fichier.shape == (*config.IMAGE_SIZE, 3)
    assert np.array_equal(depuis_fichier, depuis_image)


def test_normalisation_appliquee_une_seule_fois(mini_dataset):
    """Aucune double normalisation : les bornes restent dans [-1, 1]."""
    dossier, _ = mini_dataset
    _, val_ds, _ = charger_ensembles(dossier, batch_size=8, graine=42)

    lot, _ = next(iter(val_ds))
    valeurs = lot.numpy()
    assert valeurs.min() >= -1.0 - 1e-6
    assert valeurs.max() <= 1.0 + 1e-6


def test_entrainement_normalise_dans_le_meme_intervalle(mini_dataset):
    """Après augmentation + normalisation, les valeurs restent dans [-1, 1]."""
    dossier, _ = mini_dataset
    train_ds, _, _ = charger_ensembles(dossier, batch_size=8, graine=42)
    lot_images, etiquettes = next(iter(train_ds))

    assert lot_images.shape[1:] == (*config.IMAGE_SIZE, 3)
    assert float(lot_images.numpy().min()) >= -1.0 - 1e-5
    assert float(lot_images.numpy().max()) <= 1.0 + 1e-5
    assert etiquettes.shape[-1] == config.NUM_CLASSES  # encodage one-hot


def test_augmentation_seulement_pour_l_entrainement():
    """L'augmentation n'est pas incluse dans le modèle sauvegardé."""
    augmentation = construire_augmentation()
    assert all("Random" in couche.__class__.__name__ for couche in augmentation.layers)


# --- métadonnées du fine-tuning --------------------------------------------
def _petit_modele():
    entree = keras.layers.Input(shape=(8, 8, 3))
    x = keras.layers.Conv2D(4, 3, padding="same")(entree)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(4, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    base = keras.Model(entree, x, name="mobilenetv2_test")
    modele = keras.Sequential(
        [base, keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(3, activation="softmax")]
    )
    return modele, base


def test_resume_fine_tuning_distingue_drapeau_et_poids():
    """Les couches BatchNormalization (et les activations) ne sont pas comptées à tort."""
    modele, base = _petit_modele()
    base.trainable = True
    for couche in base.layers:
        couche.trainable = not isinstance(couche, keras.layers.BatchNormalization)

    detail = resume_fine_tuning(base, couches_demandees=2, learning_rate=1e-5)

    assert detail["couches_demandees"] == 2
    # Dernières couches : Conv2D (dégelée, avec poids) et BatchNormalization (gelée).
    assert detail["couches_degelees_par_drapeau"] == 1
    assert detail["couches_entrainables_avec_poids"] == 1
    assert detail["batch_normalization_gelees"] == 2
    assert detail["batch_normalization_gelees_dans_demiere_couches"] == 1


def test_decrire_architecture_lit_le_nombre_de_classes():
    """La description est calculée depuis le modèle, jamais codée en dur."""
    modele, base = _petit_modele()
    description = decrire_architecture(modele, base)

    assert description["nombre_classes"] == 3
    assert "Dense(3, softmax)" in description["resume"]
    assert description["couches_base"] == len(base.layers)
    assert description["parametres_totaux"] == int(modele.count_params())
