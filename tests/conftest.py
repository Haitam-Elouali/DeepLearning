"""Fixtures partagées par les tests."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))


@pytest.fixture()
def image_feuille():
    """Image PNG synthétique (dégradé vert) pour tester le téléversement."""
    image = Image.new("RGB", (256, 256), (34, 139, 34))
    for x in range(0, 256, 4):
        image.putpixel((x, x), (20 + (x % 40), 120, 30))
    tampon = io.BytesIO()
    image.save(tampon, format="PNG")
    tampon.seek(0)
    return tampon
