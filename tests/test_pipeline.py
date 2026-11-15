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


def _jeu_reduit(dossier: Path, par_classe: int = 2) -> dict[str, list[Path]]:
    """Crée un mini-dataset (train/validation/test) avec les 38 classes."""
    from PIL import Image

    images: dict[str, list[Path]] = {}
    for indice, classe in enumerate(config.CLASS_NAMES):
        fichiers = []
        for split in ("train", "validation", "test"):
            dossier_classe = dossier / split / classe
            dossier_classe.mkdir(parents=True, exist_ok=True)
            for numero in range(par_classe):
                chemin = dossier_classe / f"{numero}.png"
                couleur = ((indice * 7) % 256, (numero * 90) % 256, (indice * 13) % 256)
                Image.new("RGB", (256, 256), couleur).save(chemin, format="PNG")
                fichiers.append(chemin)
        images[classe] = fichiers
    return images


@pytest.fixture(scope="module")
def mini_dataset(tmp_path_factory):
    dossier = tmp_path_factory.mktemp("mini")
    images = _jeu_reduit(dossier)
    return dossier, images


def test_validation_et_test_utilisent_le_preraitement_d_inference(mini_dataset):
    """Les tenseurs de validation sont identiques à ceux de l'application."""
    dossier, images = mini_dataset
    _, val_ds, test_ds = charger_ensembles(dossier, batch_size=8, graine=42)

    attendus = {
        classe: [preparer_pour_modele(chemin) for chemin in fichiers]
        for classe, fichiers in images.items()
    }

    for dataset in (val_ds, test_ds):
        lot_images, _ = next(iter(dataset))
        for image in lot_images.numpy():
            distances = min(
                float(np.max(np.abs(image - attendu)))
                for tenseurs in attendus.values()
                for attendu in tenseurs
            )
            assert distances < 1e-4, f"écart de prétraitement : {distances}"


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
