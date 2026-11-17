"""Tests de cohérence des résultats expérimentaux.

Vérifie que les métriques publiées (JSON, résumé Markdown, README, fiche) sont
bien celles des prédictions réelles du modèle sur l'ensemble de test, et que le
nombre d'images correspond au manifeste du dataset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import config

RACINE = Path(__file__).resolve().parent.parent
EVALUATION = config.METRICS_JSON_PATH
PREDICTIONS = config.PREDICTIONS_DIR / "predictions_test.csv"
MANIFESTE = config.PROCESSED_DATA_DIR / "dataset_manifest.json"


def _charger_csv(chemin: Path) -> list[dict[str, str]]:
    with chemin.open(newline="", encoding="utf-8") as fichier:
        return list(csv.DictReader(fichier, delimiter=";"))


def _pourcent(valeur: float, decimales: int = 2) -> str:
    return f"{valeur * 100:.{decimales}f} %".replace(".", ",")


@pytest.fixture(scope="module")
def evaluation() -> dict:
    if not EVALUATION.exists():
        pytest.skip("metrics.json absent")
    return json.loads(EVALUATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def predictions() -> list[dict[str, str]]:
    if not PREDICTIONS.exists():
        pytest.skip("predictions_test.csv absent")
    return _charger_csv(PREDICTIONS)


def test_metriques_recalculees_depuis_les_predictions(evaluation, predictions):
    """L'accuracy publiée est exactement celle des prédictions enregistrées."""
    corrects = sum(int(ligne["correct"]) for ligne in predictions)
    exactitude = corrects / len(predictions)
    assert exactitude == pytest.approx(evaluation["global"]["accuracy"], abs=1e-9)
    assert evaluation["global"]["nb_erreurs"] == len(predictions) - corrects
    assert evaluation["global"]["nb_images"] == len(predictions)


def test_nombre_d_images_de_test_correspond_au_manifeste(evaluation, predictions):
    if not MANIFESTE.exists():
        pytest.skip("manifeste absent")
    manifeste = json.loads(MANIFESTE.read_text(encoding="utf-8"))
    assert manifeste["totaux"]["test"] == len(predictions)
    assert evaluation["global"]["nb_images"] == manifeste["totaux"]["test"]


def test_supports_par_classe_correspondent_au_manifeste(evaluation):
    if not MANIFESTE.exists():
        pytest.skip("manifeste absent")
    manifeste = json.loads(MANIFESTE.read_text(encoding="utf-8"))
    for classe, valeurs in evaluation["par_classe"].items():
        assert valeurs["support"] == manifeste["comptages"]["test"][classe]


def test_matrice_de_confusion_coherente(evaluation, predictions):
    """Chaque ligne de la matrice correspond aux prédictions de la classe réelle."""
    classes = evaluation["classes"]
    matrice = evaluation["matrice_confusion"]
    attendu = {classe: [0] * len(classes) for classe in classes}
    for ligne in predictions:
        attendu[ligne["classe_reelle"]][classes.index(ligne["classe_predite"])] += 1

    for index, classe in enumerate(classes):
        assert matrice[index] == attendu[classe]


def test_matrice_diagonale_egale_aux_bien_classees(evaluation):
    for index, valeurs in enumerate(evaluation["par_classe"].values()):
        assert evaluation["matrice_confusion"][index][index] == valeurs["bien_classees"]


def test_historique_couvre_les_45_epochs():
    if not config.HISTORY_PATH.exists() or not config.TRAINING_CONFIG_PATH.exists():
        pytest.skip("historique absent")
    historique = json.loads(config.HISTORY_PATH.read_text(encoding="utf-8"))
    configuration = json.loads(config.TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    assert len(historique["loss"]) == configuration["epochs_total"]
    assert configuration["epochs_total"] == config.EPOCHS + config.FINE_TUNE_EPOCHS
    assert len(historique["val_accuracy"]) == len(historique["loss"])


def test_configuration_imbriquee_est_a_jour(evaluation):
    """metrics.json embarque la configuration complète de l'entraînement."""
    if not config.TRAINING_CONFIG_PATH.exists():
        pytest.skip("training_config.json absent")
    configuration = json.loads(config.TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    assert evaluation["configuration"] == configuration


def test_resultats_publies_dans_le_readme(evaluation):
    """Les chiffres du README sont ceux du fichier de métriques."""
    texte = (RACINE / "README.md").read_text(encoding="utf-8")
    global_ = evaluation["global"]
    attendus = [
        _pourcent(global_["accuracy"]),
        _pourcent(global_["precision_macro"]),
        _pourcent(global_["recall_macro"]),
        _pourcent(global_["f1_macro"]),
    ]
    manquants = [valeur for valeur in attendus if valeur not in texte]
    assert not manquants, f"valeurs absentes du README : {manquants}"


def test_fiche_experimentale_utilise_les_memes_chiffres(evaluation):
    fiche = RACINE / "docs" / "fiche" / "fiche_experimentale.md"
    if not fiche.exists():
        pytest.skip("fiche absente")
    texte = fiche.read_text(encoding="utf-8")
    assert _pourcent(evaluation["global"]["accuracy"]) in texte


def test_empreinte_sha256_du_modele(evaluation):
    """Les métriques identifient exactement le fichier de modèle évalué."""
    import hashlib

    if not config.MODEL_PATH.exists():
        pytest.skip("modèle absent")
    digest = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
    assert evaluation["modele_sha256"] == digest
    assert evaluation["modele_fichier"] == "models/best_model.keras"
