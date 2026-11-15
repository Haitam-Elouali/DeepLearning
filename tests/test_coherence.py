"""Tests de cohérence globale du projet (38 classes, ordre, métadonnées).

Ces tests ne chargent pas TensorFlow : ils vérifient les fichiers de
configuration et les métadonnées produites par l'entraînement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config

RACINE = Path(__file__).resolve().parent.parent
CLASSES_ATTENDUES = 38


def _json(chemin: Path) -> dict:
    return json.loads(Path(chemin).read_text(encoding="utf-8"))


# --- configuration -----------------------------------------------------------
def test_le_projet_compte_38_classes():
    assert config.NUM_CLASSES == CLASSES_ATTENDUES
    assert len(config.CLASS_NAMES) == CLASSES_ATTENDUES


def test_les_noms_de_classes_sont_uniques():
    assert len(set(config.CLASS_NAMES)) == len(config.CLASS_NAMES)


def test_les_classes_suivent_le_format_culture_maladie():
    """Chaque nom est « Culture___Maladie » (séparateur unique)."""
    for classe in config.CLASS_NAMES:
        assert "___" in classe, classe
        culture = classe.split("___")[0]
        assert culture and not culture.startswith("_")


def test_quatorze_cultures():
    cultures = {classe.split("___")[0] for classe in config.CLASS_NAMES}
    assert len(cultures) == 14, sorted(cultures)


def test_libelles_francais_pointent_sur_des_classes_existantes():
    assert set(config.CLASSES_FR) <= set(config.CLASS_NAMES)


# --- fichiers d'ordre des classes -------------------------------------------
def test_class_names_json_correspond_a_config():
    chemin = config.CLASS_NAMES_PATH
    if not chemin.exists():
        pytest.skip("data/class_names.json absent (préparation non exécutée)")
    assert _json(chemin) == config.CLASS_NAMES


def test_manifeste_utilise_le_meme_ordre_de_classes():
    chemin = config.PROCESSED_DATA_DIR / "dataset_manifest.json"
    if not chemin.exists():
        pytest.skip("manifeste absent")
    manifeste = _json(chemin)
    assert manifeste["classes"] == config.CLASS_NAMES
    assert set(manifeste["comptages"]) == {"train", "validation", "test"}


def test_training_config_decrit_le_modele_reel():
    chemin = config.TRAINING_CONFIG_PATH
    if not chemin.exists():
        pytest.skip("models/training_config.json absent")
    configuration = _json(chemin)
    assert configuration["classes"] == config.CLASS_NAMES
    assert f"Dense({config.NUM_CLASSES}, softmax)" in configuration["modele"]
    architecture = configuration["architecture"]
    assert architecture["nombre_classes"] == config.NUM_CLASSES
    assert architecture["couches_base"] > 0


def test_metadonnees_fine_tuning_coherentes():
    chemin = config.TRAINING_CONFIG_PATH
    if not chemin.exists():
        pytest.skip("models/training_config.json absent")
    phase2 = _json(chemin)["phase2_fine_tuning"]
    detail = phase2["fine_tuning"]
    assert detail["couches_demandees"] == config.FINE_TUNE_LAYERS
    # Les BatchNormalization restent gelées : moins de couches réellement
    # entraînables que de couches examinées.
    assert detail["couches_degelees_par_drapeau"] <= detail["couches_examinees"]
    assert (
        detail["couches_entrainables_avec_poids"] <= detail["couches_degelees_par_drapeau"]
    )
    assert detail["batch_normalization_gelees"] == detail["batch_normalization_total"]
    assert detail["learning_rate"] == pytest.approx(config.FINE_TUNE_LEARNING_RATE)


# --- absence de références obsolètes ---------------------------------------
def test_aucune_reference_au_modele_a_quatre_classes():
    """Aucun fichier de code ou de configuration ne décrit Dense(4)."""
    motifs = ("Dense(4,", "Dense(4 ,", "quatre classes de tomate)")
    fichiers = [
        *RACINE.glob("*.py"),
        *RACINE.glob("training/*.py"),
        *RACINE.glob("utils/*.py"),
        *RACINE.glob("scripts/*.py"),
        *RACINE.glob("models/*.json"),
        *RACINE.glob("results/metrics/*.json"),
    ]
    coupables = []
    for fichier in fichiers:
        texte = fichier.read_text(encoding="utf-8", errors="ignore")
        if any(motif in texte for motif in motifs):
            coupables.append(fichier.name)
    assert not coupables, f"Références obsolètes dans : {coupables}"


def test_documentation_ne_presente_pas_le_projet_comme_un_modele_tomate():
    """Le README décrit bien 38 classes, pas un sous-ensemble tomate."""
    texte = (RACINE / "README.md").read_text(encoding="utf-8")
    assert "38 classes" in texte
    assert "<url-du-depot>" not in texte
