"""Tests de la détection globale des doublons (MD5 + dHash).

Ces tests couvrent les cas demandés : doublons dans une même classe, doublons
entre deux classes, images différentes, fichiers homonymes de contenu différent,
et détection des quasi-doublons.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import config
from training.prepare_data import (
    construire_decoupage,
    detecter_doublons,
    distance_hamming,
    empreinte_dhash,
    empreinte_md5,
)

CLASSE_A = config.CLASS_NAMES[0]
CLASSE_B = config.CLASS_NAMES[1]


def _image(chemin: Path, couleur: tuple[int, int, int], taille: tuple[int, int] = (64, 64)) -> Path:
    """Écrit une image uniforme (contenu contrôlable) et renvoie son chemin."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", taille, couleur).save(chemin, format="JPEG", quality=95)
    return chemin


def test_deux_images_identiques_meme_classe(tmp_path):
    """Deux copies identiques dans une classe : un groupe, rien n'est écarté."""
    premiere = _image(tmp_path / CLASSE_A / "a.jpg", (10, 120, 20))
    seconde = tmp_path / CLASSE_A / "copie.jpg"
    seconde.write_bytes(premiere.read_bytes())

    analyse = detecter_doublons({CLASSE_A: [premiere, seconde]}, verifier_quasi=False)

    assert len(analyse.groupes_intra) == 1
    assert analyse.conflits == []
    assert analyse.statistiques["conflits_annotation"] == 0
    # Les deux fichiers restent dans le découpage : ils seront groupés.
    assert len(analyse.images[CLASSE_A]) == 2


def test_deux_images_identiques_classes_differentes(tmp_path):
    """Image identique dans deux classes : conflit d'annotation, image écartée."""
    premiere = _image(tmp_path / CLASSE_A / "a.jpg", (200, 40, 40))
    seconde = tmp_path / CLASSE_B / "b.jpg"
    seconde.parent.mkdir(parents=True, exist_ok=True)
    seconde.write_bytes(premiere.read_bytes())

    analyse = detecter_doublons({CLASSE_A: [premiere], CLASSE_B: [seconde]}, verifier_quasi=False)

    assert len(analyse.conflits) == 1
    assert analyse.conflits[0]["classes"] == sorted([CLASSE_A, CLASSE_B])
    assert analyse.statistiques["images_ecartees_conflits"] == 2
    # Aucune classe n'est choisie arbitrairement : les deux images sont écartées.
    assert analyse.images[CLASSE_A] == []
    assert analyse.images[CLASSE_B] == []


def test_images_differentes_ne_sont_pas_des_doublons(tmp_path):
    """Deux images de contenus distincts : ni doublon exact ni conflit."""
    premiere = _image(tmp_path / CLASSE_A / "a.jpg", (0, 0, 0))
    seconde = _image(tmp_path / CLASSE_A / "b.jpg", (255, 255, 255))

    analyse = detecter_doublons({CLASSE_A: [premiere, seconde]}, verifier_quasi=False)

    assert analyse.groupes_intra == {}
    assert analyse.conflits == []
    assert len(analyse.images[CLASSE_A]) == 2


def test_homonymes_de_contenu_different(tmp_path):
    """Même nom de fichier dans deux classes mais contenus différents : pas de conflit."""
    premiere = _image(tmp_path / CLASSE_A / "feuille.jpg", (10, 200, 10))
    seconde = _image(tmp_path / CLASSE_B / "feuille.jpg", (10, 10, 200))

    analyse = detecter_doublons({CLASSE_A: [premiere], CLASSE_B: [seconde]}, verifier_quasi=False)

    assert analyse.conflits == []
    assert len(analyse.images[CLASSE_A]) == 1
    assert len(analyse.images[CLASSE_B]) == 1


def test_detection_globale_couvre_toutes_les_classes(tmp_path):
    """Un doublon entre la première et la dernière classe est bien détecté."""
    premiere = _image(tmp_path / CLASSE_A / "a.jpg", (90, 90, 90))
    derniere = tmp_path / config.CLASS_NAMES[-1] / "z.jpg"
    derniere.parent.mkdir(parents=True, exist_ok=True)
    derniere.write_bytes(premiere.read_bytes())

    analyse = detecter_doublons({CLASSE_A: [premiere], config.CLASS_NAMES[-1]: [derniere]})

    assert len(analyse.conflits) == 1
    assert analyse.statistiques["images_analysees"] == 2


def test_quasi_doublons_signales_sans_suppression(tmp_path):
    """Une copie recompressée est un quasi-doublon : signalée, jamais supprimée."""
    originale = _image(tmp_path / CLASSE_A / "originale.jpg", (40, 160, 60), taille=(128, 128))
    quasi = tmp_path / CLASSE_A / "recompressee.jpg"
    with Image.open(originale) as image:
        image.save(quasi, format="JPEG", quality=40)

    assert empreinte_md5(originale) != empreinte_md5(quasi)
    assert distance_hamming(empreinte_dhash(originale), empreinte_dhash(quasi)) <= 2

    analyse = detecter_doublons({CLASSE_A: [originale, quasi]}, verifier_quasi=True)

    assert analyse.groupes_intra == {}
    assert analyse.quasi_doublons
    assert analyse.quasi_doublons[0]["inter_classes"] is False
    assert len(analyse.images[CLASSE_A]) == 2  # rien n'est supprimé


def test_quasi_doublons_entre_classes_signales(tmp_path):
    """Un quasi-doublon inter-classes est marqué comme tel (inspection manuelle)."""
    originale = _image(tmp_path / CLASSE_A / "a.jpg", (30, 30, 30), taille=(128, 128))
    copie = tmp_path / CLASSE_B / "b.jpg"
    copie.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(originale) as image:
        image.save(copie, format="JPEG", quality=45)

    analyse = detecter_doublons({CLASSE_A: [originale], CLASSE_B: [copie]}, verifier_quasi=True)

    assert analyse.conflits == []  # ce ne sont pas des doublons exacts
    groupe = next(g for g in analyse.quasi_doublons if len(g["fichiers"]) == 2)
    assert groupe["inter_classes"] is True


def test_decoupage_ne_separe_jamais_les_doublons(tmp_path, monkeypatch):
    """Deux copies identiques atterrissent dans le même ensemble."""
    monkeypatch.setattr(config, "MIN_IMAGES_PER_SPLIT", 1)
    monkeypatch.setattr(config, "CLASS_NAMES", [CLASSE_A])
    premiere = _image(tmp_path / CLASSE_A / "a.jpg", (70, 20, 90))
    seconde = tmp_path / CLASSE_A / "copie.jpg"
    seconde.write_bytes(premiere.read_bytes())
    autres = [
        _image(tmp_path / CLASSE_A / f"autre_{index}.jpg", (index, index, index))
        for index in range(1, 12)
    ]

    analyse = detecter_doublons({CLASSE_A: [premiere, seconde, *autres]}, verifier_quasi=False)
    decoupage, _ = construire_decoupage(
        analyse.images, ratio_val=0.15, ratio_test=0.15, graine=42, groupes=analyse.groupes_intra
    )

    emplacements = {
        split: {chemin.name for chemin in decoupage[split][CLASSE_A]}
        for split in ("train", "validation", "test")
    }
    assert any(
        {"a.jpg", "copie.jpg"} <= noms for noms in emplacements.values()
    ), emplacements
    assert sum(len(noms) for noms in emplacements.values()) == len(autres) + 2


def test_empreinte_md5_stable(tmp_path):
    """Le MD5 ne dépend ni du nom ni de l'emplacement du fichier."""
    premiere = _image(tmp_path / "x" / "a.jpg", (5, 5, 5))
    seconde = tmp_path / "y" / "b.jpg"
    seconde.parent.mkdir(parents=True, exist_ok=True)
    seconde.write_bytes(premiere.read_bytes())

    assert empreinte_md5(premiere) == empreinte_md5(seconde)


def test_erreur_si_classe_absente_du_dataset(tmp_path):
    """Une classe manquante est signalée explicitement."""
    from training.prepare_data import collecter_images

    with pytest.raises(SystemExit):
        collecter_images(tmp_path / "inexistant")
