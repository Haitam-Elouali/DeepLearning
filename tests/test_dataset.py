"""Tests du découpage du dataset et de la vérification des 38 classes.

Couverture demandée par le cahier des charges :

- nombre total d'images conservé après découpage ;
- effectifs de chaque ensemble ;
- proportions réellement obtenues (70 / 15 / 15) ;
- absence de doublons exacts entre les ensembles ;
- reproductibilité du découpage (graine fixe) ;
- vérification du téléchargement sur les 38 classes (pas seulement la première).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import config
from training.prepare_data import (
    SPLITS,
    construire_decoupage,
    detecter_doublons,
    empreinte_md5,
)

#: Nombre d'images par classe dans le mini-dataset de test.
IMAGES_PAR_CLASSE = 200


def _ecrire_images(dossier: Path, nombre: int, indice: int) -> None:
    """Écrit des images différentes (dégradé + motif) pour tests."""
    dossier.mkdir(parents=True, exist_ok=True)
    for numero in range(nombre):
        tableau = np.zeros((16, 16, 3), dtype=np.uint8)
        tableau[..., 0] = (numero * 7 + indice) % 256
        tableau[..., 1] = (numero * 3) % 256
        tableau[numero % 16, :, 2] = 255
        Image.fromarray(tableau).save(dossier / f"{numero:04d}.png")


@pytest.fixture()
def petit_dataset(tmp_path, monkeypatch):
    """Mini-dataset à deux classes, sans la règle des 30 images mini."""
    monkeypatch.setattr(config, "MIN_IMAGES_PER_SPLIT", 1)
    monkeypatch.setattr(config, "CLASS_NAMES", config.CLASS_NAMES[:2])

    images: dict[str, list[Path]] = {}
    for indice, classe in enumerate(config.CLASS_NAMES):
        dossier = tmp_path / classe
        _ecrire_images(dossier, IMAGES_PAR_CLASSE, indice)
        images[classe] = sorted(dossier.iterdir())
    return images


def _decouper(images):
    analyse = detecter_doublons(images, verifier_quasi=False)
    return construire_decoupage(
        analyse.images,
        ratio_val=config.VALIDATION_SPLIT,
        ratio_test=config.TEST_SPLIT,
        graine=config.SEED,
        groupes=analyse.groupes_intra,
    )


def test_decoupage_conserve_toutes_les_images(petit_dataset):
    """Aucune image n'est perdue : total = somme des trois ensembles."""
    decoupage, details = _decouper(petit_dataset)

    for classe in config.CLASS_NAMES:
        par_split = {split: len(decoupage[split][classe]) for split in SPLITS}
        assert sum(par_split.values()) == IMAGES_PAR_CLASSE, classe
    assert details["images_utilisees"] == IMAGES_PAR_CLASSE * len(config.CLASS_NAMES)


def test_decoupage_respecte_les_proportions_70_15_15(petit_dataset):
    """Les proportions sont calculées sur le nombre réel d'images."""
    decoupage, details = _decouper(petit_dataset)

    for classe in config.CLASS_NAMES:
        total = IMAGES_PAR_CLASSE
        parts = {split: len(decoupage[split][classe]) / total for split in SPLITS}
        assert abs(parts["train"] - 0.70) <= 0.03, (classe, parts)
        assert abs(parts["validation"] - 0.15) <= 0.03, (classe, parts)
        assert abs(parts["test"] - 0.15) <= 0.03, (classe, parts)
        # 200 images : le découpage exact est 140 / 30 / 30.
        assert len(decoupage["train"][classe]) == 140
        assert len(decoupage["validation"][classe]) == 30
        assert len(decoupage["test"][classe]) == 30
        assert details["proportions_par_classe"][classe]["train"] == pytest.approx(
            0.70, abs=0.01
        )


def test_aucun_doublon_exact_entre_les_ensembles(petit_dataset):
    """Une empreinte MD5 n'apparaît que dans un seul ensemble."""
    decoupage, _ = _decouper(petit_dataset)

    splits_par_empreinte: dict[str, set[str]] = {}
    for split in SPLITS:
        for classe in config.CLASS_NAMES:
            for chemin in decoupage[split][classe]:
                splits_par_empreinte.setdefault(empreinte_md5(chemin), set()).add(split)

    partages = {
        empreinte: splits
        for empreinte, splits in splits_par_empreinte.items()
        if len(splits) > 1
    }
    assert partages == {}


def test_decoupage_reproductible(petit_dataset):
    """Deux découpages avec la graine 42 donnent exactement les mêmes fichiers."""
    premier, _ = _decouper(petit_dataset)
    second, _ = _decouper(petit_dataset)

    for split in SPLITS:
        for classe in config.CLASS_NAMES:
            noms_premier = sorted(chemin.name for chemin in premier[split][classe])
            noms_second = sorted(chemin.name for chemin in second[split][classe])
            assert noms_premier == noms_second


def test_minimum_de_30_images_signale_pour_les_classes_rares(tmp_path, monkeypatch):
    """Le minimum de 30 images/ensemble est documenté pour les classes rares."""
    monkeypatch.setattr(config, "CLASS_NAMES", config.CLASS_NAMES[:1])
    dossier = tmp_path / config.CLASS_NAMES[0]
    _ecrire_images(dossier, 100, 0)  # 100 images : 15 % = 15 < 30

    _, details = _decouper({config.CLASS_NAMES[0]: sorted(dossier.iterdir())})

    rares = details["classes_rares_ecartees_des_ratios"]
    assert len(rares) == 1
    assert rares[0]["classe"] == config.CLASS_NAMES[0]
    assert rares[0]["images"] == 100
    assert config.MIN_IMAGES_PER_SPLIT == 30


def test_verification_du_telechargement_couvre_les_38_classes(tmp_path):
    """Le contrôle du dataset compte chaque classe, pas seulement la première."""
    from training.download_dataset import verifier_dataset

    _ecrire_images(tmp_path / config.CLASS_NAMES[0], 3, 0)
    (tmp_path / config.CLASS_NAMES[1]).mkdir(parents=True)  # présente mais vide

    comptes = verifier_dataset(tmp_path)

    assert len(comptes) == config.NUM_CLASSES
    assert comptes[config.CLASS_NAMES[0]] == 3
    assert comptes[config.CLASS_NAMES[1]] == 0
    manquantes = [classe for classe, nombre in comptes.items() if nombre == 0]
    assert len(manquantes) == config.NUM_CLASSES - 1


def test_manifeste_reel_coherent():
    """Le manifeste produit sur les vraies données décrit bien 38 classes."""
    chemin = config.DATASET_MANIFEST_PATH
    if not chemin.exists():
        pytest.skip("dataset_manifest.json absent : lancez training/prepare_data.py")

    manifeste = json.loads(chemin.read_text(encoding="utf-8"))
    assert len(manifeste["classes"]) == config.NUM_CLASSES

    totaux = manifeste["totaux"]
    total = manifeste["total_general"]
    assert total == sum(totaux.values())
    assert abs(totaux["train"] / total - 0.70) <= 0.02
    assert abs(totaux["validation"] / total - 0.15) <= 0.02
    assert abs(totaux["test"] / total - 0.15) <= 0.02

    # Chaque classe conserve la totalité de ses images.
    for classe in config.CLASS_NAMES:
        par_split = sum(manifeste["comptages"][s][classe] for s in SPLITS)
        assert par_split == manifeste["images_brutes"][classe] or par_split <= manifeste[
            "images_brutes"
        ][classe]


# ---------------------------------------------------------------------------
# Téléchargement : comportement de --force
# ---------------------------------------------------------------------------
def _dataset_complet(racine: Path) -> Path:
    """Crée un dataset factice complet (38 dossiers avec images)."""
    destination = racine / "PlantVillage"
    for indice, classe in enumerate(config.CLASS_NAMES):
        _ecrire_images(destination / classe, 2, indice)
    return destination


def test_dataset_complet_sans_force_ne_telecharge_pas(tmp_path):
    """Dataset complet sans --force : rien n'est téléchargé ni supprimé."""
    from training.download_dataset import etat_dataset

    destination = _dataset_complet(tmp_path)

    assert etat_dataset(destination, force=False) == "conserver"
    assert destination.is_dir()
    assert len(list(destination.iterdir())) == config.NUM_CLASSES


def test_dataset_complet_avec_force_est_remplace(tmp_path):
    """Dataset complet avec --force : l'ancien dataset est réellement supprimé."""
    from training.download_dataset import etat_dataset, preparer_remplacement

    destination = _dataset_complet(tmp_path)

    assert etat_dataset(destination, force=True) == "telecharger"
    preparer_remplacement(destination)
    assert not destination.exists()


def test_dataset_incomplet_sans_force_echoue(tmp_path):
    """Dataset incomplet sans --force : erreur explicite, pas de faux succès."""
    from training.download_dataset import etat_dataset

    destination = tmp_path / "PlantVillage"
    _ecrire_images(destination / config.CLASS_NAMES[0], 2, 0)

    with pytest.raises(SystemExit):
        etat_dataset(destination, force=False)
    # Avec --force, le remplacement est autorisé.
    assert etat_dataset(destination, force=True) == "telecharger"
