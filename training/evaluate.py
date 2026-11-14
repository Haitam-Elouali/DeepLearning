"""Évaluation du modèle sur l'ensemble de test indépendant.

Le script calcule les métriques à partir des **prédictions réelles** du modèle
sur les images du dossier ``data/processed/test`` (jamais utilisées pendant
l'entraînement) et génère les figures du rapport :

- accuracy, precision, recall, F1-score (macro et pondérés) ;
- matrice de confusion (effectifs et proportions) ;
- classification report détaillé ;
- performances par classe ;
- courbes d'apprentissage (accuracy et loss) ;
- analyse des erreurs (CSV + planche d'images mal classées).

Sorties :
    results/metrics/evaluation.json
    results/metrics/classification_report.txt
    results/metrics/evaluation_summary.md
    results/figures/{training_curves,confusion_matrix,per_class_performance,confidence_distribution}.png
    results/predictions/{predictions_test.csv,erreurs.csv,error_grid.png}

Usage :
    python training/evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config  # noqa: E402
from utils.helpers import charger_json, nom_affichage, sauvegarder_json  # noqa: E402
from utils.predictor import obtenir_predictor  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "savefig.bbox": "tight"})


def collecter_test(dossier: Path) -> tuple[list[Path], list[int], list[str]]:
    """Liste les images de test et leur étiquette réelle."""
    chemins: list[Path] = []
    etiquettes: list[int] = []
    for indice, classe in enumerate(config.CLASS_NAMES):
        dossier_classe = dossier / classe
        if not dossier_classe.is_dir():
            raise SystemExit(f"Dossier de test manquant : {dossier_classe}")
        for fichier in sorted(dossier_classe.iterdir()):
            if fichier.is_file() and fichier.suffix.lower() in config.ALLOWED_EXTENSIONS:
                chemins.append(fichier)
                etiquettes.append(indice)
    if not chemins:
        raise SystemExit("Aucune image de test trouvée. Lancez training/prepare_data.py.")
    return chemins, etiquettes, list(config.CLASS_NAMES)


def predire_tout(predictor, chemins: list[Path], batch_size: int) -> np.ndarray:
    """Prédit par lots successifs afin de limiter l'usage mémoire."""
    probabilites: list[np.ndarray] = []
    for debut in range(0, len(chemins), batch_size):
        lot = chemins[debut : debut + batch_size]
        probabilites.append(np.asarray(predictor.predire_lot(lot), dtype=np.float64))
        print(f"   {min(debut + batch_size, len(chemins))}/{len(chemins)} images traitées", end="\r")
    print(" " * 60, end="\r")
    return np.concatenate(probabilites, axis=0)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def tracer_courbes(historique: dict, configuration: dict, chemin: Path) -> None:
    """Courbes accuracy et loss avec séparation des deux phases."""
    epochs = range(1, len(historique.get("loss", [])) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, historique.get("accuracy", []), "o-", color="#15803d", label="Entraînement")
    axes[0].plot(epochs, historique.get("val_accuracy", []), "s--", color="#b91c1c", label="Validation")
    axes[0].set_title("Accuracy par époque")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, historique.get("loss", []), "o-", color="#15803d", label="Entraînement")
    axes[1].plot(epochs, historique.get("val_loss", []), "s--", color="#b91c1c", label="Validation")
    axes[1].set_title("Perte (categorical crossentropy) par époque")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    phase1 = int(configuration.get("phase1", {}).get("epochs_effectues", 0) or 0)
    if 0 < phase1 < len(historique.get("loss", [])):
        for axe in axes:
            axe.axvline(phase1 + 0.5, color="#64748b", linestyle=":", linewidth=1.2)
            axe.text(
                phase1 + 0.7,
                axe.get_ylim()[0] + (axe.get_ylim()[1] - axe.get_ylim()[0]) * 0.05,
                "fine-tuning →",
                fontsize=8,
                color="#64748b",
            )

    figure.suptitle("Courbes d'apprentissage — MobileNetV2 (Transfer Learning)", fontsize=11)
    figure.tight_layout()
    figure.savefig(chemin)
    plt.close(figure)


def tracer_matrice(matrice: np.ndarray, chemin: Path) -> None:
    """Matrice de confusion en effectifs et en proportions."""
    noms = [nom_affichage(classe) for classe in config.CLASS_NAMES]
    normalisee = matrice.astype(float) / np.maximum(matrice.sum(axis=1, keepdims=True), 1)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.heatmap(
        matrice, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[0],
        xticklabels=noms, yticklabels=noms, linewidths=0.5, linecolor="white",
    )
    axes[0].set_title("Matrice de confusion (effectifs)")
    axes[0].set_xlabel("Classe prédite")
    axes[0].set_ylabel("Classe réelle")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        normalisee, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[1],
        xticklabels=noms, yticklabels=noms, linewidths=0.5, linecolor="white", vmin=0, vmax=1,
    )
    axes[1].set_title("Matrice de confusion (proportions par classe réelle)")
    axes[1].set_xlabel("Classe prédite")
    axes[1].set_ylabel("Classe réelle")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].tick_params(axis="y", rotation=0)

    figure.suptitle("Évaluation sur l'ensemble de test", fontsize=11)
    figure.tight_layout()
    figure.savefig(chemin)
    plt.close(figure)


def tracer_performances(rapport: dict, chemin: Path) -> None:
    """Histogrammes groupés precision / recall / F1 par classe."""
    classes = list(rapport["par_classe"].keys())
    noms = [nom_affichage(classe).replace(" de la tomate", "") for classe in classes]
    metriques = ["precision", "recall", "f1_score"]
    etiquettes = ["Precision", "Recall", "F1-score"]
    couleurs = ["#166534", "#22c55e", "#0ea5e9"]

    x = np.arange(len(classes))
    largeur = 0.26
    figure, axe = plt.subplots(figsize=(10, 4.4))
    for index, (metrique, etiquette, couleur) in enumerate(zip(metriques, etiquettes, couleurs)):
        valeurs = [rapport["par_classe"][classe][metrique] for classe in classes]
        barres = axe.bar(x + (index - 1) * largeur, valeurs, largeur, label=etiquette, color=couleur)
        axe.bar_label(barres, fmt="%.2f", fontsize=7, padding=2)

    axe.set_xticks(x, noms, rotation=12)
    axe.set_ylim(0, 1.12)
    axe.set_ylabel("Score")
    axe.set_title("Performances par classe sur l'ensemble de test")
    axe.legend(ncols=3, loc="lower center")
    figure.tight_layout()
    figure.savefig(chemin)
    plt.close(figure)


def tracer_confiance(
    confiances_succes: np.ndarray, confiances_erreurs: np.ndarray, chemin: Path
) -> None:
    """Distribution de la confiance pour les prédictions correctes et erronées."""
    figure, axe = plt.subplots(figsize=(8.5, 4))
    bornes = np.linspace(0, 1, 21)
    axe.hist(confiances_succes, bins=bornes, alpha=0.75, label="Prédictions correctes",
             color="#15803d", edgecolor="white")
    if confiances_erreurs.size:
        axe.hist(confiances_erreurs, bins=bornes, alpha=0.8, label="Erreurs",
                 color="#b91c1c", edgecolor="white")
    axe.set_xlabel("Confiance du modèle (probabilité de la classe prédite)")
    axe.set_ylabel("Nombre d'images")
    axe.set_title("Confiance selon la justesse de la prédiction")
    axe.legend()
    figure.tight_layout()
    figure.savefig(chemin)
    plt.close(figure)


def planche_erreurs(erreurs: list[dict], chemin: Path, maximum: int = 12) -> None:
    """Planche d'images mal classées (vraie classe vs classe prédite)."""
    if not erreurs:
        return
    selection = erreurs[:maximum]
    colonnes = 4
    lignes = int(np.ceil(len(selection) / colonnes))
    figure, axes = plt.subplots(lignes, colonnes, figsize=(12, 3.1 * lignes))
    axes = np.atleast_1d(axes).ravel()

    for axe in axes[len(selection):]:
        axe.axis("off")

    for axe, erreur in zip(axes, selection):
        with Image.open(erreur["chemin"]) as image:
            axe.imshow(image.convert("RGB"))
        axe.axis("off")
        axe.set_title(
            f"Réel : {nom_affichage(erreur['classe_reelle'])}\n"
            f"Prédit : {nom_affichage(erreur['classe_predite'])} "
            f"({erreur['confiance'] * 100:.0f} %)",
            fontsize=7.5,
        )

    figure.suptitle("Exemples d'erreurs de classification sur le test", fontsize=11)
    figure.tight_layout()
    figure.savefig(chemin)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main() -> int:
    parseur = argparse.ArgumentParser(description="Évaluation du modèle sur le test.")
    parseur.add_argument("--data-dir", type=Path, default=config.PROCESSED_DATA_DIR)
    parseur.add_argument("--batch-size", type=int, default=16)
    arguments = parseur.parse_args()

    print("=" * 70)
    print(" ÉVALUATION SUR L'ENSEMBLE DE TEST")
    print("=" * 70)

    predictor = obtenir_predictor()
    predictor.charger()
    configuration = charger_json(config.TRAINING_CONFIG_PATH, defaut={}) or {}

    chemins, etiquettes_reelles, classes = collecter_test(arguments.data_dir / "test")
    print(f" Modèle    : {predictor.version}")
    print(f" Test      : {len(chemins)} images — {', '.join(classes)}")

    probabilites = predire_tout(predictor, chemins, arguments.batch_size)
    predictions = probabilites.argmax(axis=1)
    confiances = probabilites.max(axis=1)

    etiquettes = np.array(etiquettes_reelles)
    exactitude = float(accuracy_score(etiquettes, predictions))
    matrice = confusion_matrix(etiquettes, predictions, labels=list(range(len(classes))))
    precision, rappel, f1, support = precision_recall_fscore_support(
        etiquettes, predictions, labels=list(range(len(classes))), zero_division=0
    )
    macro = precision_recall_fscore_support(
        etiquettes, predictions, average="macro", zero_division=0
    )
    pondere = precision_recall_fscore_support(
        etiquettes, predictions, average="weighted", zero_division=0
    )

    rapport = {
        "par_classe": {
            classe: {
                "precision": float(precision[i]),
                "recall": float(rappel[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
                "bien_classees": int(matrice[i, i]),
                "confiance_moyenne": float(
                    confiances[etiquettes == i].mean() if np.any(etiquettes == i) else 0.0
                ),
            }
            for i, classe in enumerate(classes)
        }
    }

    correct = predictions == etiquettes
    erreurs_details = [
        {
            "chemin": chemins[index],
            "fichier": chemins[index].name,
            "classe_reelle": classes[etiquettes[index]],
            "classe_predite": classes[predictions[index]],
            "confiance": float(confiances[index]),
        }
        for index in np.where(~correct)[0]
    ]
    erreurs_details.sort(key=lambda element: element["confiance"], reverse=True)

    paires = Counter(
        (element["classe_reelle"], element["classe_predite"]) for element in erreurs_details
    )

    resultat = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "modele": predictor.version,
        "classes": classes,
        "global": {
            "accuracy": exactitude,
            "precision_macro": float(macro[0]),
            "recall_macro": float(macro[1]),
            "f1_macro": float(macro[2]),
            "precision_weighted": float(pondere[0]),
            "recall_weighted": float(pondere[1]),
            "f1_weighted": float(pondere[2]),
            "nb_images": int(len(etiquettes)),
            "nb_erreurs": int((~correct).sum()),
        },
        **rapport,
        "matrice_confusion": matrice.tolist(),
        "confiance": {
            "moyenne_succes": float(confiances[correct].mean()) if correct.any() else 0.0,
            "moyenne_erreurs": (
                float(confiances[~correct].mean()) if (~correct).any() else 0.0
            ),
            "mediane": float(np.median(confiances)),
        },
        "erreurs": {
            "total": int((~correct).sum()),
            "taux": float((~correct).mean()),
            "paires_les_plus_frequentes": [
                {"reelle": reelle, "predite": predite, "nombre": nombre}
                for (reelle, predite), nombre in paires.most_common(5)
            ],
        },
        "configuration": configuration,
    }

    # --- Écriture des artefacts ----------------------------------------
    for dossier in (config.FIGURES_DIR, config.METRICS_DIR, config.PREDICTIONS_DIR):
        dossier.mkdir(parents=True, exist_ok=True)

    sauvegarder_json(resultat, config.METRICS_DIR / "evaluation.json")

    rapport_texte = classification_report(
        etiquettes,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    (config.METRICS_DIR / "classification_report.txt").write_text(
        "Classification report — ensemble de test\n"
        f"Modèle : {predictor.version}\n\n" + rapport_texte,
        encoding="utf-8",
    )

    # CSV de toutes les prédictions
    with (config.PREDICTIONS_DIR / "predictions_test.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fichier:
        redacteur = csv.writer(fichier, delimiter=";")
        redacteur.writerow(["fichier", "classe_reelle", "classe_predite", "confiance", "correct"])
        for index, chemin in enumerate(chemins):
            redacteur.writerow(
                [
                    chemin.name,
                    classes[etiquettes[index]],
                    classes[predictions[index]],
                    f"{confiances[index]:.4f}",
                    int(correct[index]),
                ]
            )

    with (config.PREDICTIONS_DIR / "erreurs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fichier:
        redacteur = csv.writer(fichier, delimiter=";")
        redacteur.writerow(["fichier", "classe_reelle", "classe_predite", "confiance"])
        for erreur in erreurs_details:
            redacteur.writerow(
                [
                    erreur["fichier"],
                    erreur["classe_reelle"],
                    erreur["classe_predite"],
                    f"{erreur['confiance']:.4f}",
                ]
            )

    # --- Figures --------------------------------------------------------
    historique = charger_json(config.HISTORY_PATH, defaut={}) or {}
    if historique:
        tracer_courbes(historique, configuration, config.FIGURES_DIR / "training_curves.png")
    tracer_matrice(matrice, config.FIGURES_DIR / "confusion_matrix.png")
    tracer_performances(rapport, config.FIGURES_DIR / "per_class_performance.png")
    tracer_confiance(
        confiances[correct],
        confiances[~correct],
        config.FIGURES_DIR / "confidence_distribution.png",
    )
    planche_erreurs(erreurs_details, config.PREDICTIONS_DIR / "error_grid.png")

    # --- Résumé Markdown -------------------------------------------------
    lignes = [
        "# Résultats de l'évaluation — ensemble de test",
        "",
        f"- Modèle : `{predictor.version}`",
        f"- Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        f"- Images de test : {len(etiquettes)}",
        "",
        "## Métriques globales",
        "",
        "| Métrique | Valeur |",
        "| --- | --- |",
        f"| Accuracy | {exactitude:.4f} |",
        f"| Precision (macro) | {macro[0]:.4f} |",
        f"| Recall (macro) | {macro[1]:.4f} |",
        f"| F1-score (macro) | {macro[2]:.4f} |",
        f"| F1-score (pondéré) | {pondere[2]:.4f} |",
        f"| Images mal classées | {(~correct).sum()} / {len(etiquettes)} |",
        "",
        "## Performances par classe",
        "",
        "| Classe | Precision | Recall | F1 | Support | Bien classées |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for classe, valeurs in rapport["par_classe"].items():
        lignes.append(
            f"| {classe} | {valeurs['precision']:.4f} | {valeurs['recall']:.4f} | "
            f"{valeurs['f1_score']:.4f} | {valeurs['support']} | {valeurs['bien_classees']} |"
        )

    lignes += ["", "## Matrice de confusion", "", "| Réel \\ Prédit | " + " | ".join(classes) + " |"]
    lignes.append("| --- | " + " | ".join("---" for _ in classes) + " |")
    for index, classe in enumerate(classes):
        lignes.append(
            f"| {classe} | " + " | ".join(str(int(v)) for v in matrice[index]) + " |"
        )

    lignes += ["", "## Analyse des erreurs", ""]
    lignes.append(
        f"- Confiance moyenne sur les prédictions correctes : {resultat['confiance']['moyenne_succes']:.4f}"
    )
    lignes.append(
        f"- Confiance moyenne sur les erreurs : {resultat['confiance']['moyenne_erreurs']:.4f}"
    )
    for paire in resultat["erreurs"]["paires_les_plus_frequentes"]:
        lignes.append(
            f"- {paire['reelle']} → {paire['predite']} : {paire['nombre']} cas"
        )

    (config.METRICS_DIR / "evaluation_summary.md").write_text(
        "\n".join(lignes) + "\n", encoding="utf-8"
    )

    # --- Affichage console ----------------------------------------------
    print("\n" + "-" * 70)
    print(f" Accuracy         : {exactitude:.4f}")
    print(f" Precision (macro) : {macro[0]:.4f}")
    print(f" Recall (macro)    : {macro[1]:.4f}")
    print(f" F1-score (macro)  : {macro[2]:.4f}")
    print(f" F1-score pondéré  : {pondere[2]:.4f}")
    print("-" * 70)
    print(rapport_texte)
    print(" Matrice de confusion :\n", matrice)
    print("-" * 70)
    print(f" Erreurs : {(~correct).sum()} / {len(etiquettes)}")
    print(f" Figures : {config.FIGURES_DIR}")
    print(f" Métriques : {config.METRICS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
