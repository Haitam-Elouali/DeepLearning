"""Téléchargement du sous-ensemble « tomate » du dataset PlantVillage.

Le dataset complet (54 306 images, 38 classes, 14 cultures) n'est pas
versionné : ce script récupère uniquement les quatre dossiers utiles du dépôt
GitHub officiel, via un clone partiel (``--filter=blob:none`` + checkout
creux), afin de ne télécharger que quelques dizaines de mégaoctets.

Source  : https://github.com/spMohanty/PlantVillage-Dataset
Article : Mohanty, Hughes & Salathé (2016), Frontiers in Plant Science 7:1419.

Usage :
    python training/download_dataset.py
    python training/download_dataset.py --force
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

DEPOT = "https://github.com/spMohanty/PlantVillage-Dataset.git"
DOSSIER_BRUT = "raw/color"  # images couleur d'origine (256 x 256)


def executer(commande: list[str], dossier: Path | None = None) -> None:
    """Exécute une commande git et affiche sa progression."""
    resultat = subprocess.run(commande, cwd=dossier, text=True)
    if resultat.returncode != 0:
        raise SystemExit(f"Échec de la commande : {' '.join(commande)}")


def compter_images(dossier: Path) -> int:
    """Compte les images d'une classe."""
    if not dossier.is_dir():
        return 0
    return sum(
        1
        for fichier in dossier.iterdir()
        if fichier.is_file() and fichier.suffix.lower() in config.ALLOWED_EXTENSIONS
    )


def main() -> int:
    parseur = argparse.ArgumentParser(description="Téléchargement de PlantVillage (tomate).")
    parseur.add_argument("--destination", type=Path, default=config.RAW_DATA_DIR)
    parseur.add_argument("--force", action="store_true", help="Retélécharge même si présent.")
    arguments = parseur.parse_args()

    destination: Path = arguments.destination
    if destination.exists() and compter_images(destination / config.CLASS_NAMES[0]) > 0:
        if not arguments.force:
            print(f"Dataset déjà présent dans {destination}. Utilisez --force pour le remplacer.")
            return 0
        shutil.rmtree(destination)

    if shutil.which("git") is None:
        raise SystemExit(
            "git est requis. Installez-le (apt install git) ou téléchargez manuellement\n"
            f"les dossiers {DOSSIER_BRUT}/Tomato___* depuis {DEPOT}"
        )

    temporaire = config.BASE_DIR / ".plantvillage_tmp"
    if temporaire.exists():
        shutil.rmtree(temporaire)

    print("=" * 70)
    print(" TÉLÉCHARGEMENT DU DATASET PLANTVILLAGE (4 classes de tomate)")
    print("=" * 70)
    print(f" Dépôt : {DEPOT}")
    print(" Clone partiel : seules les images couleur des 4 classes sont récupérées.\n")

    executer(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth",
            "1",
            DEPOT,
            str(temporaire),
        ]
    )
    executer(["git", "sparse-checkout", "init", "--cone"], temporaire)
    executer(
        ["git", "sparse-checkout", "set"]
        + [f"{DOSSIER_BRUT}/{classe}" for classe in config.CLASS_NAMES],
        temporaire,
    )

    # Le dépôt utilise la branche « master ».
    for branche in ("master", "main"):
        resultat = subprocess.run(
            ["git", "checkout", branche], cwd=temporaire, capture_output=True, text=True
        )
        if resultat.returncode == 0:
            break
    else:
        raise SystemExit("Impossible de trouver la branche du dépôt (master/main).")

    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    for classe in config.CLASS_NAMES:
        source = temporaire / DOSSIER_BRUT / classe
        cible = destination / classe
        if not source.is_dir():
            raise SystemExit(f"Dossier absent dans le dépôt : {source}")
        if cible.exists():
            shutil.rmtree(cible)
        shutil.copytree(source, cible)
        nombre = compter_images(cible)
        total += nombre
        attendu = config.EXPECTED_IMAGES_PER_CLASS.get(classe)
        avertissement = "" if attendu == nombre else f" (attendu : {attendu})"
        print(f"   {classe:<28} {nombre:>5} images{avertissement}")

    shutil.rmtree(temporaire, ignore_errors=True)
    print(f"\n Total : {total} images copiées dans {destination}")

    if total < sum(config.EXPECTED_IMAGES_PER_CLASS.values()) * 0.9:
        print("\n /!\\ Le nombre d'images est inférieur à environ 90 % des valeurs de référence.")
        print("     Vérifiez votre connexion ou la version du dépôt.")

    print(
        "\n Dataset prêt. Étape suivante : python training/prepare_data.py\n"
        "\n Rappel : les images proviennent du projet PlantVillage (Penn State University)\n"
        " et sont diffusées à des fins de recherche. Citer : Mohanty, Hughes & Salathé\n"
        " (2016), Frontiers in Plant Science 7:1419. Le dépôt ne comporte pas de fichier\n"
        " de licence explicite : vérifier les conditions d'usage avant toute\n"
        " rediffusion, et ne pas inclure les images dans un dépôt public."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
