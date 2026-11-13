"""Préparation du dataset : contrôle qualité, découpage stratifié, manifeste.

Ce script transforme les images brutes de PlantVillage (``data/raw``) en trois
ensembles disjoints (``data/processed/train``, ``validation``, ``test``) et
produit un manifeste JSON documentant précisément l'opération.

Garanties anti-fuite :
- un découpage **stratifié** par classe (mêmes proportions partout) ;
- une graine aléatoire fixe (reproductibilité) ;
- les **doublons exacts** (même empreinte MD5) sont détectés puis placés dans
  le même ensemble : une image identique ne peut pas se retrouver à la fois en
  entraînement et en test ;
- les images corrompues ou illisibles sont écartées et signalées.

Usage :
    python training/prepare_data.py
    python training/prepare_data.py --max-per-class 400 --force
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Permet d'exécuter le script depuis n'importe quel répertoire.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, UnidentifiedImageError  # noqa: E402

import config  # noqa: E402
from utils.helpers import sauvegarder_json  # noqa: E402

SPLITS = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Analyse du dossier brut
# ---------------------------------------------------------------------------
def empreinte_md5(chemin: Path) -> str:
    """Calcule l'empreinte MD5 d'un fichier (détection de doublons)."""
    digest = hashlib.md5()
    with chemin.open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(65536), b""):
            digest.update(bloc)
    return digest.hexdigest()


def verifier_image(chemin: Path) -> tuple[bool, str | None, tuple[int, int] | None]:
    """Vérifie qu'un fichier est une image RGB exploitable."""
    try:
        with Image.open(chemin) as image:
            image.load()
            image = image.convert("RGB")
            return True, None, image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, str(exc), None


def collecter_images(dossier_brut: Path) -> dict[str, list[Path]]:
    """Parcourt les dossiers de classe et retourne les images valides."""
    if not dossier_brut.is_dir():
        raise SystemExit(
            f"Dossier introuvable : {dossier_brut}\n"
            "Lancez d'abord : python training/download_dataset.py"
        )

    images: dict[str, list[Path]] = {}
    for classe in config.CLASS_NAMES:
        dossier = dossier_brut / classe
        if not dossier.is_dir():
            raise SystemExit(f"Classe manquante dans le dataset : {classe}")
        fichiers = sorted(
            chemin
            for chemin in dossier.iterdir()
            if chemin.is_file() and chemin.suffix.lower() in config.ALLOWED_EXTENSIONS
        )
        images[classe] = fichiers
    return images


# ---------------------------------------------------------------------------
# Découpage stratifié sans fuite
# ---------------------------------------------------------------------------
def construire_decoupage(
    images: dict[str, list[Path]],
    ratio_val: float,
    ratio_test: float,
    graine: int,
    max_par_classe: int | None,
) -> tuple[dict[str, dict[str, list[Path]]], dict[str, object]]:
    """Construit les trois ensembles de façon stratifiée et reproductible."""
    generateur = random.Random(graine)
    decoupage: dict[str, dict[str, list[Path]]] = {split: defaultdict(list) for split in SPLITS}

    empreintes: dict[str, list[Path]] = defaultdict(list)
    doublons_exacts = 0
    total_utilise = 0

    for classe in config.CLASS_NAMES:
        fichiers = images[classe]
        # 1) On regroupe les fichiers identiques pour ne jamais les séparer.
        for chemin in fichiers:
            empreintes[empreinte_md5(chemin)].append(chemin)
        groupes = list(empreintes.values())
        doublons_exacts += len(fichiers) - len(groupes)
        empreintes.clear()

        generateur.shuffle(groupes)
        if max_par_classe is not None and len(groupes) > max_par_classe:
            groupes = groupes[:max_par_classe]

        nombre = len(groupes)
        total_utilise += sum(len(groupe) for groupe in groupes)
        n_test = max(config.MIN_IMAGES_PER_SPLIT, round(nombre * ratio_test))
        n_val = max(config.MIN_IMAGES_PER_SPLIT, round(nombre * ratio_val))
        n_train = nombre - n_val - n_test
        if n_train < config.MIN_IMAGES_PER_SPLIT:
            raise SystemExit(
                f"Pas assez d'images pour la classe {classe} avec les ratios demandés."
            )

        # 2) Chaque groupe est affecté en bloc à l'ensemble qui a le plus
        #    grand déficit par rapport à son objectif : le découpage reste
        #    stratifié (~70/15/15) sans jamais séparer des images identiques.
        objectifs = {
            "train": n_train,
            "validation": n_val,
            "test": nombre - n_train - n_val,
        }
        remplies = {split: 0 for split in SPLITS}
        for groupe in groupes:
            cible = max(SPLITS, key=lambda split: objectifs[split] - remplies[split])
            decoupage[cible][classe].extend(groupe)
            remplies[cible] += len(groupe)

    details = {"doublons_exacts": doublons_exacts, "images_utilisees": total_utilise}
    return decoupage, details


# ---------------------------------------------------------------------------
# Écriture des ensembles préparés
# ---------------------------------------------------------------------------
def ecrire_ensembles(
    decoupage: dict[str, dict[str, list[Path]]],
    dossier_sortie: Path,
    copier: bool,
) -> None:
    """Crée l'arborescence processed/{split}/{classe}/ avec liens ou copies."""
    for split in SPLITS:
        for classe in config.CLASS_NAMES:
            dossier = dossier_sortie / split / classe
            dossier.mkdir(parents=True, exist_ok=True)
            for chemin in decoupage[split][classe]:
                destination = dossier / chemin.name
                if destination.exists():
                    destination.unlink()
                if copier:
                    shutil.copy2(chemin, destination)
                else:
                    try:
                        destination.hardlink_to(chemin)
                    except (OSError, AttributeError):
                        shutil.copy2(chemin, destination)


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main() -> int:
    parseur = argparse.ArgumentParser(description="Préparation du dataset PlantVillage (tomate).")
    parseur.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    parseur.add_argument("--output-dir", type=Path, default=config.PROCESSED_DATA_DIR)
    parseur.add_argument("--val-size", type=float, default=config.VALIDATION_SPLIT)
    parseur.add_argument("--test-size", type=float, default=config.TEST_SPLIT)
    parseur.add_argument("--seed", type=int, default=config.SEED)
    parseur.add_argument(
        "--max-per-class",
        type=int,
        default=config.MAX_IMAGES_PER_CLASS,
        help="Limite le nombre d'images par classe (utile sur machine modeste).",
    )
    parseur.add_argument("--copy", action="store_true", help="Copier au lieu de lier les fichiers.")
    parseur.add_argument("--force", action="store_true", help="Écrase un découpage existant.")
    arguments = parseur.parse_args()

    if arguments.val_size + arguments.test_size >= 0.6:
        raise SystemExit("Les ratios validation + test doivent rester inférieurs à 60 %.")

    manifeste_existant = arguments.output_dir / "dataset_manifest.json"
    if manifeste_existant.exists() and not arguments.force:
        print(f"Un découpage existe déjà ({manifeste_existant}).")
        print("Relancez avec --force pour le recalculer.")
        return 0

    print("=" * 70)
    print(" PRÉPARATION DU DATASET — Détection des maladies de la tomate")
    print("=" * 70)

    images = collecter_images(arguments.raw_dir)

    # --- Contrôle qualité -------------------------------------------------
    invalides: list[dict[str, str]] = []
    tailles: Counter[str] = Counter()
    for classe, fichiers in images.items():
        valides: list[Path] = []
        for chemin in fichiers:
            ok, erreur, taille = verifier_image(chemin)
            if ok:
                valides.append(chemin)
                tailles[f"{taille[0]}x{taille[1]}"] += 1
            else:
                invalides.append({"fichier": chemin.name, "classe": classe, "erreur": erreur or ""})
        images[classe] = valides

    print("\n1. Inventaire brut")
    for classe, fichiers in images.items():
        attendu = config.EXPECTED_IMAGES_PER_CLASS.get(classe)
        print(f"   {classe:<28} {len(fichiers):>5} images (attendu : {attendu})")
    print(f"   Images illisibles écartées : {len(invalides)}")
    print(f"   Formats rencontrés : {dict(tailles.most_common(3))}")

    # --- Découpage --------------------------------------------------------
    decoupage, details = construire_decoupage(
        images,
        ratio_val=arguments.val_size,
        ratio_test=arguments.test_size,
        graine=arguments.seed,
        max_par_classe=arguments.max_per_class,
    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    ecrire_ensembles(decoupage, arguments.output_dir, arguments.copy)

    # --- Manifeste --------------------------------------------------------
    comptages = {
        split: {classe: len(decoupage[split][classe]) for classe in config.CLASS_NAMES}
        for split in SPLITS
    }
    totaux = {split: sum(comptages[split].values()) for split in SPLITS}
    total_general = sum(totaux.values())

    sauvegarder_json(config.CLASS_NAMES, config.CLASS_NAMES_PATH)

    manifeste = {
        "dataset": "PlantVillage (sous-ensemble tomate, images couleur)",
        "source": "https://github.com/spMohanty/PlantVillage-Dataset",
        "citation": "Mohanty, Hughes & Salathé (2016), Frontiers in Plant Science",
        "genere_le": datetime.now().isoformat(timespec="seconds"),
        "graine": arguments.seed,
        "ratios_demandes": {
            "train": round(1 - arguments.val_size - arguments.test_size, 4),
            "validation": arguments.val_size,
            "test": arguments.test_size,
        },
        "taille_image": list(config.IMAGE_SIZE),
        "normalisation": "x / 127.5 - 1 (convention MobileNetV2)",
        "augmentation": [
            "rotation (±5 %)",
            "zoom (±10 %)",
            "retournement horizontal",
            "variation de luminosité (±15 %)",
            "translation (±5 %)",
        ],
        "max_images_par_classe": arguments.max_per_class,
        "classes": config.CLASS_NAMES,
        "comptages": comptages,
        "totaux": totaux,
        "total_general": total_general,
        "images_brutes": {classe: len(fichiers) for classe, fichiers in images.items()},
        "doublons_exacts_ecartes_de_groupes": details["doublons_exacts"],
        "images_utilisees": details["images_utilisees"],
        "images_invalides": invalides,
        "mise_en_lien": "copie" if arguments.copy else "lien physique (hardlink)",
    }
    sauvegarder_json(manifeste, manifeste_existant)

    # --- Rapport console --------------------------------------------------
    print("\n2. Découpage stratifié")
    entete = f"   {'Classe':<28}{'Train':>8}{'Valid.':>8}{'Test':>8}{'Total':>8}"
    print(entete)
    print("   " + "-" * (len(entete) - 3))
    for classe in config.CLASS_NAMES:
        print(
            f"   {classe:<28}{comptages['train'][classe]:>8}"
            f"{comptages['validation'][classe]:>8}{comptages['test'][classe]:>8}"
            f"{sum(comptages[s][classe] for s in SPLITS):>8}"
        )
    print("   " + "-" * (len(entete) - 3))
    print(
        f"   {'TOTAL':<28}{totaux['train']:>8}{totaux['validation']:>8}"
        f"{totaux['test']:>8}{total_general:>8}"
    )

    print("\n3. Contrôles")
    print(f"   Graine aléatoire          : {arguments.seed}")
    print(f"   Doublons exacts détectés  : {details['doublons_exacts']} (placés dans le même ensemble)")
    print(f"   Fichiers illisibles       : {len(invalides)}")
    print(f"   Manifeste                 : {manifeste_existant}")
    print("\nPréparation terminée. Étape suivante : python training/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
