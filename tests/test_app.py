"""Tests de l'application Web : formulaire, validation, prédiction."""

from __future__ import annotations

import io

import pytest

from app import creer_application


@pytest.fixture()
def client():
    """Client Flask isolé (le modèle est chargé une seule fois)."""
    return creer_application().test_client()


def test_page_accueil_affiche_le_formulaire(client):
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert "Analyser l'image".encode() in reponse.data


def test_refus_sans_fichier(client):
    reponse = client.post("/analyser", data={})
    assert reponse.status_code == 400
    assert "Aucune image".encode() in reponse.data


def test_refus_extension_non_autorisee(client):
    reponse = client.post(
        "/analyser",
        data={"image": (io.BytesIO(b"GIF89a"), "feuille.gif", "image/gif")},
        content_type="multipart/form-data",
    )
    assert reponse.status_code == 400
    assert "Format non supporté".encode() in reponse.data


def test_refus_contenu_non_image(client):
    reponse = client.post(
        "/analyser",
        data={"image": (io.BytesIO(b"pas une image"), "faux.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )
    assert reponse.status_code == 400
    assert "pas une image lisible".encode() in reponse.data


def test_refus_image_trop_petite(client):
    from PIL import Image

    tampon = io.BytesIO()
    Image.new("RGB", (16, 16), (0, 128, 0)).save(tampon, format="PNG")
    tampon.seek(0)

    reponse = client.post(
        "/analyser",
        data={"image": (tampon, "mini.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert reponse.status_code == 400
    assert "trop petite".encode() in reponse.data


def test_analyse_reelle_affiche_un_resultat(client, image_feuille):
    reponse = client.post(
        "/analyser",
        data={"image": (image_feuille, "feuille.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert reponse.status_code == 200
    assert "Confiance".encode() in reponse.data
    assert "Probabilités pour chaque classe".encode() in reponse.data


def test_page_inconnue_renvoie_404(client):
    assert client.get("/page-inexistante").status_code == 404
