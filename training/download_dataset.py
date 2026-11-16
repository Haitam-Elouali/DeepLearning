"""Téléchargement des 38 classes couleur du dataset PlantVillage.

Le dataset complet (54 305 images, 38 classes, 14 cultures) n'est pas
versionné : ce script récupère uniquement les dossiers utiles du dépôt GitHub
officiel, via un clone partiel (``--filter=blob:none`` + checkout creux), afin
de ne télécharger que les images couleur des classes utilisées.

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


def verifier_dataset(destination: Path) -> dict[str, int]:
    """Compte les images valides de chacune des 38 classes attendues.

    Renvoie ``{classe: nombre d'images}`` ; une classe absente ou vide vaut 0.
    """
    return {classe: compter_images(destination / classe) for classe in config.CLASS_NAMES}


def afficher_classes_manquantes(nombre_par_classe: dict[str, int]) -> list[str]:
    """Affiche les classes absentes ou vides et renvoie leur liste."""
    manquantes = [classe for classe, nombre in nombre_par_classe.items() if nombre == 0]
    if manquantes:
        print(
            f"\n Classes manquantes ou vides "
            f"({len(manquantes)}/{len(nombre_par_classe)}) :"
        )
        for classe in manquantes:
            print(f"   - {classe}")
    return manquantes


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
    parseur = argparse.ArgumentParser(
        description=f"Téléchargement de PlantVillage ({config.NUM_CLASSES} classes)."
    )
    parseur.add_argument("--destination", type=Path, default=config.RAW_DATA_DIR)
    parseur.add_argument("--force", action="store_true", help="Retélécharge même si présent.")
    arguments = parseur.parse_args()

    destination: Path = arguments.destination

    if destination.exists():
        # Un dataset n'est complet que si les 38 dossiers existent et
        # contiennent des images : chaque classe est vérifiée, pas seulement
        # la première.
        existantes = verifier_dataset(destination)
        manquantes = afficher_classes_manquantes(existantes)
        if not manquantes:
            print(
                f"Dataset complet : {sum(existantes.values())} images dans "
                f"{len(existantes)} classes ({destination})."
            )
            print("Utilisez --force pour le retélécharger.")
            return 0
        if not arguments.force:
            raise SystemExit(
                "Dataset incomplet : le téléchargement n'est pas considéré "
                "comme terminé.\nRelancez avec --force pour récupérer "
                "les 38 classes."
            )
        print("\n Dataset incomplet : remplacement complet avec --force.")
        shutil.rmtree(destination)

    if shutil.which("git") is None:
        raise SystemExit(
            "git est requis. Installez-le (apt install git) ou téléchargez manuellement\n"
            f"les dossiers {DOSSIER_BRUT}/<Classe> depuis {DEPOT} ({config.NUM_CLASSES} classes)"
        )

    temporaire = config.BASE_DIR / ".plantvillage_tmp"
    if temporaire.exists():
        shutil.rmtree(temporaire)

    print("=" * 70)
    print(" TÉLÉCHARGEMENT DU DATASET PLANTVILLAGE (38 classes, 14 cultures)")
    print("=" * 70)
    print(f" Dépôt : {DEPOT}")
    print(" Clone partiel : seules les images couleur des 38 classes sont récupérées.\n")

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
        print(f"   {classe:<50} {nombre:>5} images")

    shutil.rmtree(temporaire, ignore_errors=True)

    # Contrôle final : les 38 classes doivent être présentes et non vides.
    controle = verifier_dataset(destination)
    manquantes = afficher_classes_manquantes(controle)
    if manquantes:
        raise SystemExit("Téléchargement incomplet : " + ", ".join(manquantes))
    print(
        f"\n Total : {sum(controle.values())} images dans {len(controle)} "
        f"classes copiées dans {destination}"
    )

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
