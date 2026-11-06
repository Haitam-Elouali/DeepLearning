"""Tests des utilitaires : validation de fichiers, noms, formatage."""

from __future__ import annotations

import io

import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage

import config
from utils.helpers import (
    charger_infos_maladies,
    classe_est_saine,
    enregistrer_upload,
    extension_autorisee,
    formater_date,
    formater_pourcentage,
    infos_classe,
    nom_affichage,
    nom_fichier_securise,
    tronquer,
)


@pytest.mark.parametrize(
    "nom,attendu",
    [
        ("feuille.jpg", True),
        ("feuille.JPEG", True),
        ("photo.png", True),
        ("document.pdf", False),
        ("archive.zip", False),
        ("sans_extension", False),
        ("", False),
    ],
)
def test_extension_autorisee(nom, attendu):
    assert extension_autorisee(nom) is attendu


def test_nom_fichier_securise_supprime_les_chemins():
    nom = nom_fichier_securise("../../etc/passwd.jpg")
    assert "/" not in nom and ".." not in nom
    assert nom.endswith(".jpg")


def test_nom_fichier_securise_conserve_les_accents_simplifies():
    nom = nom_fichier_securise("feuille été 2026.jpg")
    assert " " in nom
    assert nom.endswith(".jpg")


def _fichier(nom: str, contenu: bytes, type_mime: str = "image/png") -> FileStorage:
    return FileStorage(stream=io.BytesIO(contenu), filename=nom, content_type=type_mime)


def test_enregistrer_upload_valide(image_feuille, dossier_uploads):
    contenu = image_feuille.getvalue()
    resultat = enregistrer_upload(_fichier("ma feuille.png", contenu))

    assert resultat["chemin"].exists()
    assert resultat["nom_stocke"].endswith(".jpg")
    assert len(resultat["nom_stocke"]) == 36  # UUID + extension
    assert resultat["nom_original"] == "ma feuille.png"
    with Image.open(resultat["chemin"]) as image:
        assert image.mode == "RGB"


def test_enregistrer_upload_refuse_une_extension_invalide(dossier_uploads):
    with pytest.raises(ValueError, match="Format"):
        enregistrer_upload(_fichier("feuille.gif", b"GIF89a"))


def test_enregistrer_upload_refuse_un_faux_type_mime(image_feuille, dossier_uploads):
    with pytest.raises(ValueError, match="Type"):
        enregistrer_upload(
            _fichier("feuille.jpg", image_feuille.getvalue(), type_mime="application/pdf")
        )


def test_enregistrer_upload_refuse_un_contenu_invalide(dossier_uploads):
    with pytest.raises(ValueError, match="image"):
        enregistrer_upload(_fichier("faux.jpg", b"pas une image"))


def test_enregistrer_upload_refuse_une_image_trop_petite(dossier_uploads):
    tampon = io.BytesIO()
    Image.new("RGB", (10, 10), (0, 255, 0)).save(tampon, format="PNG")
    with pytest.raises(ValueError, match="petite"):
        enregistrer_upload(_fichier("mini.png", tampon.getvalue()))


def test_formater_pourcentage():
    assert formater_pourcentage(0.9234) == "92,3 %"
    assert formater_pourcentage(None) == "—"


def test_formater_date():
    assert formater_date("2026-11-06T14:35:00") == "06/11/2026 à 14:35"
    assert formater_date(None) == "—"


def test_tronquer():
    assert tronquer("abc", 10) == "abc"
    assert len(tronquer("a" * 50, 10)) <= 10


def test_infos_classes_completes():
    infos = charger_infos_maladies()
    assert set(infos) == set(config.CLASS_NAMES)
    for classe in config.CLASS_NAMES:
        assert infos[classe]["nom_fr"]
        assert infos[classe]["symptomes"]
        assert infos_classe(classe)["description"]


def test_nom_affichage_et_classe_saine():
    assert nom_affichage("Tomato___healthy") == "Feuille saine"
    assert nom_affichage("Tomato___Late_blight") == "Mildiou de la tomate"
    assert classe_est_saine("Tomato___healthy")
    assert not classe_est_saine("Tomato___Early_blight")


def test_infos_classe_inconnue_ne_plante_pas():
    infos = infos_classe("Tomato___Inexistante")
    assert infos["resume"]
    assert infos["couleur"].startswith("#")
