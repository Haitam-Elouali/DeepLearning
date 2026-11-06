"""Fixtures partagées par les tests.

Les tests d'application utilisent une base SQLite et un dossier d'uploads
temporaires : la base réelle et les images de l'utilisateur ne sont jamais
modifiées.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

import config  # noqa: E402
from utils import database  # noqa: E402


@pytest.fixture()
def base_temporaire(tmp_path, monkeypatch):
    """Base SQLite isolée pour chaque test."""
    chemin = tmp_path / "test_history.db"
    monkeypatch.setattr(config, "DATABASE_PATH", chemin)
    database.initialiser_base(chemin)
    return chemin


@pytest.fixture()
def dossier_uploads(tmp_path, monkeypatch):
    """Dossier d'uploads isolé pour chaque test."""
    dossier = tmp_path / "uploads"
    dossier.mkdir()
    monkeypatch.setattr(config, "UPLOAD_DIR", dossier)
    return dossier


@pytest.fixture()
def image_feuille():
    """Image PNG synthétique (dégradé vert) pour tester le téléversement."""
    image = Image.new("RGB", (256, 256), (34, 139, 34))
    for x in range(256):
        for y in range(0, 256, 8):
            image.putpixel((x, y), (20 + (x % 40), 120, 30))
    tampon = io.BytesIO()
    image.save(tampon, format="PNG")
    tampon.seek(0)
    return tampon


@pytest.fixture(scope="session")
def image_de_test():
    """Une image réelle du jeu de test (si le dataset est présent)."""
    dossier = config.PROCESSED_DATA_DIR / "test"
    if not dossier.is_dir():
        pytest.skip("Jeu de test absent : exécuter training/prepare_data.py")
    for classe in config.CLASS_NAMES:
        candidats = sorted((dossier / classe).glob("*.jpg")) + sorted(
            (dossier / classe).glob("*.JPG")
        )
        if candidats:
            return candidats[0]
    pytest.skip("Aucune image de test disponible.")


@pytest.fixture(scope="session")
def application():
    """Application Flask avec le vrai modèle (nécessite un modèle entraîné)."""
    if not config.MODEL_PATH.exists():
        pytest.skip("Modèle absent : exécuter training/train.py")
    from app import creer_application

    appli = creer_application()
    appli.config.update(TESTING=True)
    return appli
