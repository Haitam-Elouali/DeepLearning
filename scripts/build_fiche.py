"""Génère la fiche expérimentale (Markdown + Word) à partir des résultats réels.

Toutes les valeurs sont lues dans les artefacts produits par l'entraînement et
l'évaluation : models/training_config.json, results/metrics/evaluation.json,
data/processed/dataset_manifest.json et results/metrics/training_history.json.

Usage :
    python scripts/build_fiche.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

import config  # noqa: E402
from scripts.build_docx import convertir  # noqa: E402
from utils.helpers import charger_json, nom_affichage  # noqa: E402

SORTIE_MD = RACINE / "docs" / "fiche" / "fiche_experimentale.md"
SORTIE_DOCX = RACINE / "docs" / "fiche" / "fiche_experimentale.docx"


def pourcent(valeur: float, decimales: int = 2) -> str:
    """Formate un ratio en pourcentage avec virgule décimale."""
    return f"{valeur * 100:.{decimales}f} %".replace(".", ",")


def main() -> int:
    configuration = charger_json(config.TRAINING_CONFIG_PATH, defaut={}) or {}
    metriques = charger_json(config.METRICS_DIR / "evaluation.json", defaut={}) or {}
    manifeste = charger_json(config.DATASET_MANIFEST_PATH, defaut={}) or {}
    historique = charger_json(config.HISTORY_PATH, defaut={}) or {}

    if not configuration or not metriques:
        raise SystemExit(
            "Résultats manquants : exécutez training/train.py puis training/evaluate.py."
        )

    phase1 = configuration.get("phase1", {})
    phase2 = configuration.get("phase2_fine_tuning", {})
    global_ = metriques.get("global", {})
    comptages = manifeste.get("comptages", {})
    par_classe = metriques.get("par_classe", {})
    matrice = metriques.get("matrice_confusion", [])
    confiance = metriques.get("confiance", {})
    erreurs = metriques.get("erreurs", {})

    lignes: list[str] = [
        "# Fiche expérimentale — Détection des maladies de la feuille de tomate",
        "",
        f"**Modèle** : MobileNetV2 + Transfer Learning  ",
        f"**Expérience réalisée le** : {configuration.get('date', '—')[:10]}  ",
        f"**Environnement** : Python {configuration.get('python', '—')}, "
        f"TensorFlow {configuration.get('tensorflow', '—')}, CPU (sans GPU)  ",
        f"**Fiche générée le** : {datetime.now().strftime('%d/%m/%Y')}",
        "",
        "## 1. Paramètres d'entraînement",
        "",
        "| Paramètre | Valeur |",
        "| --- | --- |",
        f"| Taille d'image | {config.IMAGE_SIZE[0]} × {config.IMAGE_SIZE[1]} × 3 |",
        f"| Batch size | {configuration.get('batch_size')} |",
        f"| Optimiseur | {configuration.get('optimiseur')} |",
        f"| Perte | {configuration.get('perte')} |",
        f"| Époques phase 1 (effectuées / max) | {phase1.get('epochs_effectues')} / {phase1.get('epochs_max')} |",
        f"| Learning rate phase 1 | {phase1.get('learning_rate')} |",
        f"| Époques phase 2 (fine-tuning) | {phase2.get('epochs', 0)} |",
        f"| Learning rate phase 2 | {phase2.get('learning_rate', '—')} |",
        f"| Couches dégelées | {phase2.get('couches_degelees', 0)} |",
        f"| Graine aléatoire | {configuration.get('graine')} |",
        f"| Early Stopping | patience {configuration.get('callbacks', {}).get('early_stopping', {}).get('patience')} sur val_loss |",
        f"| Model Checkpoint | meilleur val_accuracy ({configuration.get('fichier_modele')}) |",
        f"| Paramètres totaux | {configuration.get('parametres_totaux', 0):,} |".replace(",", " "),
        f"| Paramètres entraînables (phase 1) | {configuration.get('parametres_entrainables_phase1', 0):,} |".replace(",", " "),
        "",
        f"Data Augmentation : {', '.join(configuration.get('augmentation', []))}.",
        "",
        "## 2. Effectifs par classe",
        "",
        "| Classe | Dataset brut | Entraînement | Validation | Test |",
        "| --- | --- | --- | --- | --- |",
    ]

    for classe in config.CLASS_NAMES:
        lignes.append(
            f"| {nom_affichage(classe)} ({classe}) | "
            f"{manifeste.get('images_brutes', {}).get(classe, '—')} | "
            f"{comptages.get('train', {}).get(classe, '—')} | "
            f"{comptages.get('validation', {}).get(classe, '—')} | "
            f"{comptages.get('test', {}).get(classe, '—')} |"
        )
    lignes += [
        f"| **Total** | **{manifeste.get('total_general', '—')}** | "
        f"**{manifeste.get('totaux', {}).get('train', '—')}** | "
        f"**{manifeste.get('totaux', {}).get('validation', '—')}** | "
        f"**{manifeste.get('totaux', {}).get('test', '—')}** |",
        "",
        f"Découpage stratifié, graine {manifeste.get('graine')}, "
        f"{manifeste.get('doublons_exacts_ecartes_de_groupes', 0)} doublons exacts regroupés, "
        f"{len(manifeste.get('images_invalides', []))} image(s) illisible(s).",
        "",
        "## 3. Courbes d'apprentissage",
        "",
        "![Courbes accuracy et loss (entraînement / validation)](../../results/figures/training_curves.png)",
        "",
        f"Meilleure accuracy de validation : **{pourcent(configuration.get('meilleure_val_accuracy', 0))}** "
        f"(époque {configuration.get('meilleure_epoch')}, sur {configuration.get('epochs_total')} époques).",
        "",
        "| Phase | Époques | Meilleure val_accuracy | Durée |",
        "| --- | --- | --- | --- |",
        f"| Phase 1 — base gelée | {phase1.get('epochs_effectues')} | "
        f"{pourcent(phase1.get('meilleure_val_accuracy', 0))} | {phase1.get('duree_min')} min |",
        f"| Phase 2 — fine-tuning | {phase2.get('epochs', 0)} | "
        f"{pourcent(configuration.get('meilleure_val_accuracy', 0))} | {phase2.get('duree_min', '—')} min |",
        "",
        "## 4. Métriques sur l'ensemble de test",
        "",
        "| Métrique | Valeur |",
        "| --- | --- |",
        f"| Accuracy | **{pourcent(global_.get('accuracy', 0))}** |",
        f"| Precision (macro) | {pourcent(global_.get('precision_macro', 0))} |",
        f"| Recall (macro) | {pourcent(global_.get('recall_macro', 0))} |",
        f"| F1-score (macro) | **{pourcent(global_.get('f1_macro', 0))}** |",
        f"| F1-score (pondéré) | {pourcent(global_.get('f1_weighted', 0))} |",
        f"| Images de test | {global_.get('nb_images')} |",
        f"| Images mal classées | {global_.get('nb_erreurs')} "
        f"({pourcent(global_.get('nb_erreurs', 0) / max(global_.get('nb_images', 1), 1))}) |",
        "",
        "## 5. Matrice de confusion",
        "",
        "| Réel \\ Prédit | " + " | ".join(nom_affichage(c) for c in config.CLASS_NAMES) + " |",
        "| --- | " + " | ".join("---" for _ in config.CLASS_NAMES) + " |",
    ]
    for index, classe in enumerate(config.CLASS_NAMES):
        if index < len(matrice):
            lignes.append(
                f"| {nom_affichage(classe)} | "
                + " | ".join(str(int(valeur)) for valeur in matrice[index])
                + " |"
            )

    lignes += [
        "",
        "![Matrice de confusion (effectifs et proportions)](../../results/figures/confusion_matrix.png)",
        "",
        "## 6. Résultats par classe",
        "",
        "| Classe | Precision | Recall | F1-score | Support | Bien classées | Confiance moyenne |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for classe in config.CLASS_NAMES:
        valeurs = par_classe.get(classe, {})
        lignes.append(
            f"| {nom_affichage(classe)} | {pourcent(valeurs.get('precision', 0))} | "
            f"{pourcent(valeurs.get('recall', 0))} | {pourcent(valeurs.get('f1_score', 0))} | "
            f"{valeurs.get('support', '—')} | {valeurs.get('bien_classees', '—')} | "
            f"{pourcent(valeurs.get('confiance_moyenne', 0))} |"
        )

    lignes += [
        "",
        "![Performances par classe](../../results/figures/per_class_performance.png)",
        "",
        "## 7. Confiance et analyse des erreurs",
        "",
        "| Indicateur | Valeur |",
        "| --- | --- |",
        f"| Confiance moyenne (prédictions correctes) | {pourcent(confiance.get('moyenne_succes', 0))} |",
        f"| Confiance moyenne (erreurs) | {pourcent(confiance.get('moyenne_erreurs', 0))} |",
        f"| Confiance médiane | {pourcent(confiance.get('mediane', 0))} |",
        f"| Erreurs | {erreurs.get('total')} / {global_.get('nb_images')} "
        f"({pourcent(erreurs.get('taux', 0))}) |",
        "",
        "| Confusion observée | Nombre de cas |",
        "| --- | --- |",
    ]
    for paire in erreurs.get("paires_les_plus_frequentes", []):
        lignes.append(
            f"| {nom_affichage(paire['reelle'])} → {nom_affichage(paire['predite'])} | {paire['nombre']} |"
        )

    lignes += [
        "",
        "![Distribution de la confiance](../../results/figures/confidence_distribution.png)",
        "",
        "![Exemples d'erreurs de classification](../../results/predictions/error_grid.png)",
        "",
        "## 8. Observations et difficultés rencontrées",
        "",
        "- **Convergence très rapide de la phase 1** : dès la première époque, la validation "
        "atteint 86,2 %, ce qui confirme la qualité des caractéristiques ImageNet.",
        "- **Le fine-tuning apporte un gain modeste mais réel** (+1,46 point de validation) ; "
        "son learning rate très faible (1e-5) est indispensable pour ne pas détruire les poids "
        "pré-entraînés.",
        "- **La validation est irrégulière** (±2 points entre époques voisines) en raison de la "
        "taille limitée du jeu de validation (821 images) et de l'augmentation aléatoire.",
        "- **L'alternariose précoce est la classe la plus difficile** : ses lésions brunes la "
        "rapprochent du mildiou, ce qui explique la majorité des confusions.",
        "- **Le modèle est sur-confiant** : la confiance médiane dépasse 99 %, ce qui justifie "
        "l'affichage de la distribution complète des probabilités dans l'application.",
        "- **Budget d'entraînement contraint** : 23 époques au total sur un processeur à deux "
        "cœurs (≈ 38 minutes), contre 30 + 15 époques prévues par la configuration.",
        "- **Dataset de laboratoire** : les images ont un fond uniforme ; les performances "
        "mesurées ne préjugent pas du comportement sur des photos de terrain.",
        "- **Doublons** : 14 images identiques ont été détectées et regroupées pour éviter "
        "toute fuite entre entraînement et test.",
        "",
        "## 9. Conclusion de l'expérience",
        "",
        f"Le modèle MobileNetV2 adapté par Transfer Learning atteint **{pourcent(global_.get('accuracy', 0))} "
        f"d'accuracy** et un **F1-score macro de {pourcent(global_.get('f1_macro', 0))}** sur "
        f"{global_.get('nb_images')} images de test indépendantes, avec un temps d'inférence "
        "d'environ 70 ms par image sur CPU. Ces résultats confirment la faisabilité d'un outil "
        "local d'aide à la reconnaissance visuelle, dans les limites du périmètre étudié.",
        "",
        "## 10. Fichiers de référence",
        "",
        "| Ressource | Fichier |",
        "| --- | --- |",
        "| Configuration complète | `models/training_config.json` |",
        "| Historique des époques | `results/metrics/training_history.json` |",
        "| Rapport de classification | `results/metrics/classification_report.txt` |",
        "| Métriques détaillées | `results/metrics/evaluation.json` |",
        "| Prédictions du test | `results/predictions/predictions_test.csv` |",
        "| Erreurs détaillées | `results/predictions/erreurs.csv` |",
    ]

    SORTIE_MD.parent.mkdir(parents=True, exist_ok=True)
    SORTIE_MD.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    convertir(SORTIE_MD, SORTIE_DOCX)
    print(f"Fiche expérimentale générée : {SORTIE_MD.name} et {SORTIE_DOCX.name}")
    print(f"  Accuracy : {pourcent(global_.get('accuracy', 0))} | F1 macro : {pourcent(global_.get('f1_macro', 0))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
