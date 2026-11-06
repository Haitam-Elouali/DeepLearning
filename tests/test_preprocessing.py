"""Tests du prétraitement des images."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import config
from utils.preprocessing import (
    charger_image_rgb,
    image_vers_tableau,
    normaliser,
    preparer_pour_modele,
    redimensionner,
)


def test_charger_image_convertit_en_rgb(tmp_path):
    chemin = tmp_path / "rgba.png"
    Image.new("RGBA", (300, 200), (10, 200, 10, 128)).save(chemin)

    image = charger_image_rgb(chemin)

    assert image.mode == "RGB"
    assert image.size == (300, 200)


def test_charger_image_refuse_un_fichier_non_image(tmp_path):
    chemin = tmp_path / "faux.jpg"
    chemin.write_text("ceci n'est pas une image", encoding="utf-8")

    with pytest.raises(ValueError):
        charger_image_rgb(chemin)


def test_redimensionnement_a_la_taille_du_modele(image_feuille):
    image = charger_image_rgb(image_feuille)
    redimensionnee = redimensionner(image)
    assert redimensionnee.size == config.IMAGE_SIZE


def test_normalisation_dans_intervalle_moins_un_un(image_feuille):
    tenseur = preparer_pour_modele(image_feuille)
    assert tenseur.shape == (*config.IMAGE_SIZE, 3)
    assert tenseur.dtype == np.float32
    assert tenseur.min() >= -1.0 - 1e-5
    assert tenseur.max() <= 1.0 + 1e-5


def test_normalisation_equivalente_a_mobilenetv2(image_feuille):
    """Vérifie que le prétraitement d'inférence == prétraitement Keras."""
    keras = pytest.importorskip("tensorflow.keras")
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    image = charger_image_rgb(image_feuille)
    tableau = image_vers_tableau(redimensionner(image))
    attendu = preprocess_input(tableau.copy())

    np.testing.assert_allclose(normaliser(tableau), attendu, rtol=1e-6, atol=1e-6)


def test_lot_de_plusieurs_images(image_feuille):
    import io

    from utils.preprocessing import preparer_lot

    donnees = image_feuille.getvalue()
    lot = preparer_lot([io.BytesIO(donnees), io.BytesIO(donnees)])
    assert lot.shape == (2, *config.IMAGE_SIZE, 3)
