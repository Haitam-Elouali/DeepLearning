"""Inférence en ligne de commande avec le modèle entraîné.

Utile pour vérifier rapidement le comportement du modèle sur de nouvelles
images, sans passer par l'application Web.

Exemples :
    python training/predict.py data/processed/test/Tomato___Late_blight/xxxx.jpg
    python training/predict.py --dossier mes_photos/ --json resultats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from utils.helpers import nom_affichage  # noqa: E402
from utils.predictor import PredictionError, obtenir_predictor  # noqa: E402


def analyser(chemin: Path, predictor) -> dict:
    """Prédit une image et renvoie un dictionnaire prêt à afficher."""
    resultat = predictor.predire(chemin)
    return {
        "fichier": chemin.name,
        "classe": resultat["classe"],
        "nom_fr": nom_affichage(resultat["classe"]),
        "confiance": resultat["confiance"],
        "probabilites": {
            nom_affichage(classe): round(score, 4)
            for classe, score in sorted(
                resultat["probabilites"].items(), key=lambda element: element[1], reverse=True
            )
        },
        "duree_ms": resultat["duree_ms"],
    }


def main() -> int:
    parseur = argparse.ArgumentParser(description="Prédiction d'une ou plusieurs images.")
    parseur.add_argument("images", nargs="*", type=Path, help="Chemins des images à analyser.")
    parseur.add_argument("--dossier", type=Path, help="Analyser toutes les images d'un dossier.")
    parseur.add_argument("--json", type=Path, help="Écrire les résultats dans un fichier JSON.")
    arguments = parseur.parse_args()

    chemins: list[Path] = list(arguments.images)
    if arguments.dossier:
        chemins += sorted(
            fichier
            for fichier in arguments.dossier.iterdir()
            if fichier.is_file() and fichier.suffix.lower() in config.ALLOWED_EXTENSIONS
        )

    if not chemins:
        parseur.error("Indiquez au moins une image ou un dossier (--dossier).")

    try:
        predictor = obtenir_predictor()
        predictor.charger()
    except PredictionError as exc:
        raise SystemExit(f"Modèle indisponible : {exc}")

    resultats = []
    for chemin in chemins:
        if not chemin.is_file():
            print(f"Fichier ignoré (introuvable) : {chemin}")
            continue
        try:
            resultat = analyser(chemin, predictor)
        except (ValueError, PredictionError) as exc:
            print(f"Échec pour {chemin.name} : {exc}")
            continue

        resultats.append(resultat)
        print(f"\n{resultat['fichier']}")
        print(f"  Classe prédite : {resultat['nom_fr']} ({resultat['confiance'] * 100:.1f} %)")
        print(f"  Inférence      : {resultat['duree_ms']} ms")
        for nom, proba in resultat["probabilites"].items():
            barre = "█" * int(round(proba * 30))
            print(f"    {nom:<28} {proba * 100:5.1f} % {barre}")

    if arguments.json and resultats:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nRésultats écrits dans {arguments.json}")

    return 0 if resultats else 1


if __name__ == "__main__":
    raise SystemExit(main())
