"""Tests de la couche d'accès à la base SQLite."""

from __future__ import annotations

import re

import pytest

from utils import database


def _ajouter(database_path, classe="Tomato___healthy", confiance=0.97):
    return database.ajouter_analyse(
        nom_stocke="a" * 32 + ".jpg",
        nom_original="feuille.jpg",
        classe_predite=classe,
        confiance=confiance,
        probabilites={classe: confiance, "Tomato___Late_blight": 1 - confiance},
        modele="best_model.keras",
        duree_ms=42,
        chemin=database_path,
    )


def test_ajout_et_lecture(base_temporaire):
    identifiant = _ajouter(base_temporaire)

    analyse = database.obtenir_analyse(identifiant, chemin=base_temporaire)

    assert analyse is not None
    assert analyse["classe_predite"] == "Tomato___healthy"
    assert analyse["confiance"] == 0.97
    assert analyse["confiance_pct"] == 97.0
    assert analyse["probabilites"]["Tomato___Late_blight"] == pytest.approx(0.03)
    assert analyse["duree_ms"] == 42
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4} à \d{2}:\d{2}", analyse["date_affichage"])


def test_liste_triee_par_recence(base_temporaire):
    premier = _ajouter(base_temporaire)
    second = _ajouter(base_temporaire, classe="Tomato___Late_blight")

    analyses = database.lister_analyses(chemin=base_temporaire)

    assert [a["id"] for a in analyses] == [second, premier]
    assert database.compter_analyses(chemin=base_temporaire) == 2


def test_suppression(base_temporaire):
    identifiant = _ajouter(base_temporaire)

    assert database.supprimer_analyse(identifiant, chemin=base_temporaire) is True
    assert database.obtenir_analyse(identifiant, chemin=base_temporaire) is None
    assert database.supprimer_analyse(identifiant, chemin=base_temporaire) is False


def test_statistiques(base_temporaire):
    _ajouter(base_temporaire, confiance=0.90)
    _ajouter(base_temporaire, classe="Tomato___Late_blight", confiance=0.80)
    _ajouter(base_temporaire, classe="Tomato___Late_blight", confiance=0.70)

    stats = database.statistiques(chemin=base_temporaire)

    assert stats["total"] == 3
    assert stats["semaine"] == 3
    assert stats["par_classe"]["Tomato___Late_blight"] == 2
    assert abs(stats["confiance_moyenne"] - 0.8) < 1e-9
    assert stats["derniere_analyse"] is not None


def test_statistiques_vides(base_temporaire):
    stats = database.statistiques(chemin=base_temporaire)
    assert stats["total"] == 0
    assert stats["confiance_moyenne"] is None
    assert stats["par_classe"] == {}
    assert stats["derniere_analyse"] is None
