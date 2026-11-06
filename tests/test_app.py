"""Tests fonctionnels de l'application Flask (routes et API).

Les tests utilisent le vrai modèle entraîné : les prédictions ne sont pas
simulées. Ils sont ignorés si le modèle ou le jeu de test est absent.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

import config
from utils import database


@pytest.fixture()
def client(application, base_temporaire, dossier_uploads):
    """Client de test avec base et uploads isolés."""
    database.initialiser_base(base_temporaire)
    return application.test_client()


def test_page_accueil(client):
    reponse = client.get("/")
    assert reponse.status_code == 200
    corps = reponse.get_data(as_text=True)
    assert "Tableau de bord" in corps
    assert "MobileNetV2" in corps


@pytest.mark.parametrize("route", ["/analyse", "/historique", "/maladies", "/a-propos"])
def test_pages_repondent(client, route):
    reponse = client.get(route)
    assert reponse.status_code == 200


def test_sonde_sante(client):
    reponse = client.get("/sante")
    assert reponse.status_code == 200
    donnees = reponse.get_json()
    assert donnees["statut"] == "ok"
    assert donnees["modele_charge"] is True
    assert donnees["classes"] == len(config.CLASS_NAMES)


def test_api_refuse_une_requete_sans_fichier(client):
    reponse = client.post("/api/analyser", data={})
    assert reponse.status_code == 400
    assert reponse.get_json()["succes"] is False


def test_api_refuse_un_faux_fichier(client):
    donnees = {"image": (io.BytesIO(b"pas une image"), "feuille.jpg")}
    reponse = client.post("/api/analyser", data=donnees, content_type="multipart/form-data")
    assert reponse.status_code == 400
    assert "image" in reponse.get_json()["erreur"].lower()


def test_api_refuse_une_extension_non_autorisee(client):
    donnees = {"image": (io.BytesIO(b"GIF89a"), "feuille.gif")}
    reponse = client.post("/api/analyser", data=donnees, content_type="multipart/form-data")
    assert reponse.status_code == 400


def test_analyse_complete_puis_historique_puis_suppression(client, image_de_test, base_temporaire):
    """Parcours complet : prédiction réelle → historique → détail → suppression."""
    with image_de_test.open("rb") as fichier:
        donnees = {"image": (io.BytesIO(fichier.read()), image_de_test.name)}
        reponse = client.post(
            "/api/analyser", data=donnees, content_type="multipart/form-data"
        )

    assert reponse.status_code == 200, reponse.get_data(as_text=True)
    contenu = reponse.get_json()
    assert contenu["succes"] is True

    analyse = contenu["analyse"]
    assert analyse["classe"] in config.CLASS_NAMES
    assert 0.0 <= analyse["confiance"] <= 1.0
    assert analyse["image_url"].startswith("/image/")
    assert len(contenu["probabilites"]) == len(config.CLASS_NAMES)
    # les probabilités forment une distribution
    assert abs(sum(p["valeur"] for p in contenu["probabilites"]) - 1.0) < 1e-4

    identifiant = analyse["id"]
    assert database.obtenir_analyse(identifiant, chemin=base_temporaire) is not None

    # Page d'historique et page de détail
    assert client.get("/historique").status_code == 200
    detail = client.get(f"/analyse/{identifiant}")
    assert detail.status_code == 200
    assert analyse["classe_nom"] in detail.get_data(as_text=True)

    # Image servie avec un nom valide, refusée sinon
    nom_stocke = analyse["image_url"].rsplit("/", 1)[-1]
    assert client.get(f"/image/{nom_stocke}").status_code == 200
    assert client.get("/image/../../app.py").status_code == 404
    assert client.get("/image/invalide.jpg").status_code == 404

    # Suppression
    suppression = client.delete(f"/api/historique/{identifiant}")
    assert suppression.status_code == 200
    assert database.obtenir_analyse(identifiant, chemin=base_temporaire) is None
    assert not (config.UPLOAD_DIR / nom_stocke).exists()

    # Une seconde suppression renvoie 404
    assert client.delete(f"/api/historique/{identifiant}").status_code == 404


def test_page_detail_inexistante(client):
    assert client.get("/analyse/999999").status_code == 404


def test_api_statistiques(client):
    reponse = client.get("/api/statistiques")
    assert reponse.status_code == 200
    donnees = reponse.get_json()["statistiques"]
    assert "total" in donnees
    assert len(donnees["classes"]) == len(config.CLASS_NAMES)


def test_fichier_trop_volumineux_refuse(client, monkeypatch):
    """Un dépassement de taille est refusé (413) sans analyse."""
    monkeypatch.setitem(client.application.config, "MAX_CONTENT_LENGTH", 2048)
    # Image bruitée : incompressible, donc nettement plus grosse que la limite.
    image = Image.effect_noise((400, 400), 120).convert("RGB")
    tampon = io.BytesIO()
    image.save(tampon, format="PNG")
    donnees = {"image": (io.BytesIO(tampon.getvalue()), "grande.png")}

    reponse = client.post("/api/analyser", data=donnees, content_type="multipart/form-data")

    assert reponse.status_code in (400, 413)
