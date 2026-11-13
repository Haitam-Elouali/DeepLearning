"""Tests des utilitaires : validation d'image, noms de classes, formatage."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage

from utils.helpers import (
    charger_json,
    classe_est_saine,
    extension_autorisee,
    formater_pourcentage,
    nom_affichage,
    sauvegarder_json,
    valider_image,
)


def _fichier(nom: str, contenu: bytes, type_mime: str = "image/png") -> FileStorage:
    return FileStorage(stream=io.BytesIO(contenu), filename=nom, content_type=type_mime)


def _png(taille: tuple[int, int] = (256, 256)) -> bytes:
    tampon = io.BytesIO()
    Image.new("RGB", taille, (30, 120, 40)).save(tampon, format="PNG")
    return tampon.getvalue()


# --- extension ---------------------------------------------------------------
@pytest.mark.parametrize("nom", ["feuille.jpg", "feuille.JPEG", "feuille.png"])
def test_extension_autorisee(nom):
    assert extension_autorisee(nom) is True


@pytest.mark.parametrize("nom", ["feuille.gif", "feuille", None, ""])
def test_extension_refusee(nom):
    assert extension_autorisee(nom) is False


# --- validation d'image ------------------------------------------------------
def test_valider_image_renvoie_une_image_rgb():
    image = valider_image(_fichier("feuille.png", _png()))
    assert image.mode == "RGB"
    assert image.size == (256, 256)


def test_valider_image_sans_fichier():
    with pytest.raises(ValueError, match="Aucune image"):
        valider_image(None)


def test_valider_image_refuse_une_extension_invalide():
    with pytest.raises(ValueError, match="Format non supporté"):
        valider_image(_fichier("feuille.gif", b"GIF89a", "image/gif"))


def test_valider_image_refuse_un_faux_type_mime():
    with pytest.raises(ValueError, match="Type de fichier"):
        valider_image(_fichier("feuille.png", _png(), "application/pdf"))


def test_valider_image_refuse_un_contenu_invalide():
    with pytest.raises(ValueError, match="pas une image lisible"):
        valider_image(_fichier("faux.jpg", b"pas une image", "image/jpeg"))


def test_valider_image_refuse_une_image_trop_petite():
    with pytest.raises(ValueError, match="trop petite"):
        valider_image(_fichier("mini.png", _png((16, 16))))


def test_valider_image_refuse_une_image_trop_grande():
    with pytest.raises(ValueError, match="trop grande"):
        valider_image(_fichier("geante.png", _png((4200, 32))))


# --- affichage ---------------------------------------------------------------
def test_nom_affichage_utilise_les_libelles_francais():
    assert nom_affichage("Tomato___Late_blight") == "Mildiou"


def test_nom_affichage_repli_sur_le_nom_brut():
    assert nom_affichage("Tomato___Inconnue") == "Inconnue"


def test_classe_est_saine():
    assert classe_est_saine("Tomato___healthy") is True
    assert classe_est_saine("Tomato___Leaf_Mold") is False


def test_formater_pourcentage():
    assert formater_pourcentage(0.9583, 1) == "95,8 %"
    assert formater_pourcentage(None) == "—"


# --- JSON --------------------------------------------------------------------
def test_charger_et_sauvegarder_json(tmp_path):
    chemin = tmp_path / "donnees.json"
    sauvegarder_json({"a": 1}, chemin)
    assert charger_json(chemin) == {"a": 1}
    assert charger_json(tmp_path / "absent.json", defaut={}) == {}


def test_charger_json_invalide(tmp_path):
    chemin = tmp_path / "casse.json"
    chemin.write_text("{invalide", encoding="utf-8")
    assert charger_json(chemin, defaut=None) is None
