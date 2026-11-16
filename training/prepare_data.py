"""Préparation du dataset : contrôle qualité, doublons, découpage, manifeste.

Ce script transforme les images brutes de PlantVillage (``data/raw``) en trois
ensembles disjoints (``data/processed/train``, ``validation``, ``test``) et
produit un manifeste JSON documentant précisément l'opération.

Garanties anti-fuite :

- **détection globale des doublons** : les empreintes MD5 sont calculées sur
  l'ensemble du dataset (toutes classes confondues) *avant* le découpage ;
- les doublons exacts d'une même classe sont regroupés et restent dans le même
  ensemble : une image identique ne peut pas être à la fois en entraînement et
  en test ;
- les images **strictement identiques mais étiquetées dans deux classes
  différentes** sont des conflits d'annotation : elles sont écartées du
  découpage et signalées pour vérification manuelle (aucune classe n'est
  choisie arbitrairement) ;
- les **quasi-doublons** (empreinte perceptuelle dHash) sont détectés et
  signalés à titre informatif, mais jamais supprimés automatiquement ;
- un découpage **stratifié** par classe (mêmes proportions partout) ;
- une graine aléatoire fixe (reproductibilité) ;
- les images corrompues ou illisibles sont écartées et signalées.

Usage :
    python training/prepare_data.py
    python training/prepare_data.py --force
    python training/prepare_data.py --analyse-doublons   # rapport seul, sans split
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Permet d'exécuter le script depuis n'importe quel répertoire.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

import config  # noqa: E402
from utils.helpers import sauvegarder_json  # noqa: E402

SPLITS = ("train", "validation", "test")

#: Distance de Hamming maximale entre deux dHash pour parler de quasi-doublon.
SEUIL_QUASI_DOUBLON = 2


# ---------------------------------------------------------------------------
# Empreintes
# ---------------------------------------------------------------------------
def empreinte_md5(chemin: Path) -> str:
    """Calcule l'empreinte MD5 d'un fichier (détection de doublons exacts)."""
    digest = hashlib.md5()
    with chemin.open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(65536), b""):
            digest.update(bloc)
    return digest.hexdigest()


def empreinte_dhash(chemin: Path) -> int:
    """Calcule une empreinte perceptuelle dHash 64 bits d'une image.

    L'image est réduite en 9 x 8 niveaux de gris, puis chaque pixel est comparé
    à son voisin de droite : le résultat est insensible aux variations de
    compression et de taille, mais pas à une transformation géométrique.
    """
    with Image.open(chemin) as image:
        gris = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = np.asarray(gris, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    empreinte = 0
    for bit in bits:
        empreinte = (empreinte << 1) | int(bit)
    return empreinte


def distance_hamming(premiere: int, seconde: int) -> int:
    """Nombre de bits différents entre deux empreintes perceptuelles."""
    return bin(premiere ^ seconde).count("1")


def verifier_image(chemin: Path) -> tuple[bool, str | None, tuple[int, int] | None]:
    """Vérifie qu'un fichier est une image RGB exploitable."""
    try:
        with Image.open(chemin) as image:
            image.load()
            image = image.convert("RGB")
            return True, None, image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, str(exc), None


# ---------------------------------------------------------------------------
# Inventaire
# ---------------------------------------------------------------------------
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
# Doublons
# ---------------------------------------------------------------------------
@dataclass
class AnalyseDoublons:
    """Résultat de la détection globale des doublons."""

    #: Images conservées pour le découpage, par classe.
    images: dict[str, list[Path]] = field(default_factory=dict)
    #: Empreinte MD5 -> fichiers identiques d'une même classe (regroupés).
    groupes_intra: dict[str, list[Path]] = field(default_factory=dict)
    #: Images identiques présentes dans plusieurs classes (à vérifier à la main).
    conflits: list[dict[str, object]] = field(default_factory=list)
    #: Groupes de quasi-doublons (informatif, rien n'est supprimé).
    quasi_doublons: list[dict[str, object]] = field(default_factory=list)
    #: Compteurs pour le rapport.
    statistiques: dict[str, int] = field(default_factory=dict)

    def rapport(self) -> dict[str, object]:
        """Version sérialisable (JSON) de l'analyse."""
        return {
            "genere_le": datetime.now().isoformat(timespec="seconds"),
            "methode": {
                "doublons_exacts": "empreinte MD5 sur l'ensemble du dataset",
                "quasi_doublons": (
                    f"empreinte perceptuelle dHash 64 bits, distance de Hamming <= "
                    f"{SEUIL_QUASI_DOUBLON} (informatif : aucune suppression)"
                ),
                "seuil_quasi_doublon": SEUIL_QUASI_DOUBLON,
            },
            "statistiques": self.statistiques,
            "conflits_annotation": self.conflits,
            "doublons_intra_classe": [
                {
                    "empreinte": empreinte,
                    "classe": chemins[0].parent.name if chemins else "",
                    "fichiers": [chemin.name for chemin in chemins],
                }
                for empreinte, chemins in self.groupes_intra.items()
            ],
            "quasi_doublons": self.quasi_doublons,
        }


def detecter_doublons(
    images: dict[str, list[Path]], verifier_quasi: bool = True
) -> AnalyseDoublons:
    """Détecte les doublons exacts (globalement) puis les quasi-doublons.

    L'analyse porte sur **toutes les classes à la fois** : le dictionnaire des
    empreintes n'est jamais réinitialisé au changement de classe.
    """
    analyse = AnalyseDoublons()

    # 1) Empreintes MD5 globales -------------------------------------------
    par_empreinte: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for classe, fichiers in images.items():
        for chemin in fichiers:
            par_empreinte[empreinte_md5(chemin)].append((classe, chemin))

    images_conservees: dict[str, list[Path]] = {classe: [] for classe in images}
    total_images = 0
    ecartees = 0

    for empreinte, membres in par_empreinte.items():
        total_images += len(membres)
        classes = sorted({classe for classe, _ in membres})
        if len(classes) > 1:
            # Image identique étiquetée dans plusieurs classes : on ne tranche
            # pas, on écarte et on signale pour vérification manuelle.
            analyse.conflits.append(
                {
                    "empreinte": empreinte,
                    "classes": classes,
                    "fichiers": [
                        {"classe": classe, "fichier": chemin.name} for classe, chemin in membres
                    ],
                }
            )
            ecartees += len(membres)
            continue

        classe = classes[0]
        chemins = [chemin for _, chemin in membres]
        if len(chemins) > 1:
            analyse.groupes_intra[empreinte] = chemins
        images_conservees[classe].extend(chemins)

    for classe in images_conservees:
        images_conservees[classe].sort()

    analyse.images = images_conservees
    analyse.statistiques = {
        "images_analysees": total_images,
        "images_conservees": total_images - ecartees,
        "images_ecartees_conflits": ecartees,
        "groupes_doublons_exacts": len(analyse.groupes_intra),
        "images_dupliquees_intra_classe": sum(
            len(chemins) - 1 for chemins in analyse.groupes_intra.values()
        ),
        "conflits_annotation": len(analyse.conflits),
    }

    # 2) Quasi-doublons (dHash) --------------------------------------------
    if verifier_quasi:
        analyse.quasi_doublons = _chercher_quasi_doublons(images_conservees)
        analyse.statistiques["groupes_quasi_doublons"] = len(analyse.quasi_doublons)
        analyse.statistiques["images_quasi_dupliquees"] = sum(
            len(groupe["fichiers"]) - 1 for groupe in analyse.quasi_doublons
        )
    return analyse


def _chercher_quasi_doublons(
    images: dict[str, list[Path]], maximum_exemples: int = 200
) -> list[dict[str, object]]:
    """Regroupe les images dont les dHash sont à faible distance de Hamming.

    Aucune image n'est supprimée : le résultat sert uniquement à l'inspection.

    L'indexation par blocs de 16 bits évite la comparaison quadratique : deux
    empreintes de 64 bits à distance <= 2 partagent forcément au moins deux de
    leurs quatre blocs (principe des tiroirs), seuls ces candidats sont testés.
    """
    par_empreinte: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for classe, fichiers in images.items():
        for chemin in fichiers:
            par_empreinte[empreinte_dhash(chemin)].append((classe, chemin))

    empreintes = list(par_empreinte)
    index_blocs: dict[tuple[int, int], list[int]] = defaultdict(list)
    for empreinte in empreintes:
        for numero in range(4):
            index_blocs[(numero, (empreinte >> (16 * numero)) & 0xFFFF)].append(empreinte)

    # Union-find sur les empreintes proches (ou identiques).
    parent = {empreinte: empreinte for empreinte in empreintes}

    def racine(empreinte: int) -> int:
        while parent[empreinte] != empreinte:
            parent[empreinte] = parent[parent[empreinte]]
            empreinte = parent[empreinte]
        return empreinte

    def unir(premiere: int, seconde: int) -> None:
        racine_a, racine_b = racine(premiere), racine(seconde)
        if racine_a != racine_b:
            parent[racine_b] = racine_a

    for empreinte in empreintes:
        candidates: set[int] = set()
        for numero in range(4):
            candidates.update(index_blocs[(numero, (empreinte >> (16 * numero)) & 0xFFFF)])
        for candidate in candidates:
            if candidate != empreinte and distance_hamming(empreinte, candidate) <= SEUIL_QUASI_DOUBLON:
                unir(empreinte, candidate)

    composantes: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for empreinte in empreintes:
        composantes[racine(empreinte)].extend(par_empreinte[empreinte])

    groupes = [
        _groupe_quasi(membres) for membres in composantes.values() if len(membres) > 1
    ]
    groupes.sort(key=lambda groupe: len(groupe["fichiers"]), reverse=True)
    return groupes[:maximum_exemples]


def _groupe_quasi(membres: list[tuple[str, Path]]) -> dict[str, object]:
    """Met en forme un groupe de quasi-doublons."""
    return {
        "classes": sorted({classe for classe, _ in membres}),
        "inter_classes": len({classe for classe, _ in membres}) > 1,
        "fichiers": [{"classe": classe, "fichier": chemin.name} for classe, chemin in membres],
    }


# ---------------------------------------------------------------------------
# Découpage stratifié sans fuite
# ---------------------------------------------------------------------------
def construire_decoupage(
    images: dict[str, list[Path]],
    ratio_val: float,
    ratio_test: float,
    graine: int,
    groupes: dict[str, list[Path]] | None = None,
) -> tuple[dict[str, dict[str, list[Path]]], dict[str, object]]:
    """Construit les trois ensembles de façon stratifiée et reproductible.

    ``groupes`` associe une empreinte MD5 aux fichiers identiques : ils sont
    affectés en bloc au même ensemble pour ne jamais séparer des doublons.
    """
    generateur = random.Random(graine)
    decoupage: dict[str, dict[str, list[Path]]] = {split: defaultdict(list) for split in SPLITS}
    proportions_par_classe: dict[str, dict[str, float]] = {}
    classes_rares: list[dict[str, object]] = []
    groupes = groupes or {}

    # Empreinte de chaque fichier conservé, pour reconstituer les groupes.
    empreinte_du_fichier = {
        chemin: empreinte
        for empreinte, chemins in groupes.items()
        for chemin in chemins
    }

    total_utilise = 0
    for classe in config.CLASS_NAMES:
        fichiers = images[classe]
        # 1) Regroupement des fichiers identiques de cette classe.
        blocs: dict[str, list[Path]] = defaultdict(list)
        for chemin in fichiers:
            blocs[empreinte_du_fichier.get(chemin, f"unique:{chemin}")].append(chemin)
        liste_blocs = list(blocs.values())
        generateur.shuffle(liste_blocs)

        # Les objectifs sont exprimés en NOMBRE D'IMAGES : un bloc de doublons
        # peut en contenir plusieurs ; compter les blocs fausserait les
        # proportions. On additionne donc les images.
        nombre_images = sum(len(bloc) for bloc in liste_blocs)
        total_utilise += nombre_images

        n_test = max(config.MIN_IMAGES_PER_SPLIT, round(nombre_images * ratio_test))
        n_val = max(config.MIN_IMAGES_PER_SPLIT, round(nombre_images * ratio_val))
        n_train = nombre_images - n_val - n_test
        if n_train < config.MIN_IMAGES_PER_SPLIT:
            raise SystemExit(
                f"Pas assez d'images pour la classe {classe} avec les ratios demandés "
                f"({nombre_images} images, minimum {config.MIN_IMAGES_PER_SPLIT} "
                "par ensemble)."
            )

        # 2) Chaque bloc est affecté à l'ensemble qui a le plus grand déficit
        #    relatif : le découpage reste stratifié (~70/15/15) et les doublons
        #    exacts ne sont jamais séparés.
        objectifs = {"train": n_train, "validation": n_val, "test": n_test}
        remplies = {split: 0 for split in SPLITS}
        for bloc in liste_blocs:
            cible = max(
                SPLITS,
                key=lambda split: (objectifs[split] - remplies[split])
                / max(objectifs[split], 1),
            )
            decoupage[cible][classe].extend(bloc)
            remplies[cible] += len(bloc)

        # 3) Proportions réellement obtenues : elles sont enregistrées dans le
        #    manifeste, notamment pour les classes rares où le minimum de
        #    30 images par ensemble domine.
        obtenues = {split: len(decoupage[split][classe]) for split in SPLITS}
        proportions = {
            split: (obtenues[split] / nombre_images if nombre_images else 0.0)
            for split in SPLITS
        }
        proportions_par_classe[classe] = proportions
        if (
            abs(proportions["validation"] - ratio_val) > 0.03
            or abs(proportions["test"] - ratio_test) > 0.03
        ):
            classes_rares.append(
                {
                    "classe": classe,
                    "images": nombre_images,
                    "part_train": round(proportions["train"], 4),
                    "part_validation": round(proportions["validation"], 4),
                    "part_test": round(proportions["test"], 4),
                    "raison": (
                        f"minimum de {config.MIN_IMAGES_PER_SPLIT} images par ensemble "
                        "appliqué à une classe peu représentée"
                    ),
                }
            )

    details: dict[str, object] = {
        "images_utilisees": total_utilise,
        "proportions_par_classe": proportions_par_classe,
        "classes_rares_ecartees_des_ratios": classes_rares,
    }
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
    parseur = argparse.ArgumentParser(
        description=f"Préparation du dataset PlantVillage ({config.NUM_CLASSES} classes)."
    )
    parseur.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    parseur.add_argument("--output-dir", type=Path, default=config.PROCESSED_DATA_DIR)
    parseur.add_argument("--val-size", type=float, default=config.VALIDATION_SPLIT)
    parseur.add_argument("--test-size", type=float, default=config.TEST_SPLIT)
    parseur.add_argument("--seed", type=int, default=config.SEED)
    parseur.add_argument("--copy", action="store_true", help="Copier au lieu de lier les fichiers.")
    parseur.add_argument("--force", action="store_true", help="Écrase un découpage existant.")
    parseur.add_argument(
        "--analyse-doublons",
        action="store_true",
        help="Calcule seulement le rapport de doublons (aucun découpage écrit).",
    )
    parseur.add_argument(
        "--sans-quasi-doublons",
        action="store_true",
        help="Ignore l'analyse perceptuelle (plus rapide, moins complet).",
    )
    arguments = parseur.parse_args()

    if arguments.val_size + arguments.test_size >= 0.6:
        raise SystemExit("Les ratios validation + test doivent rester inférieurs à 60 %.")

    manifeste_existant = arguments.output_dir / "dataset_manifest.json"
    if manifeste_existant.exists() and not arguments.force and not arguments.analyse_doublons:
        print(f"Un découpage existe déjà ({manifeste_existant}).")
        print("Relancez avec --force pour le recalculer.")
        return 0

    print("=" * 70)
    print(f" PRÉPARATION DU DATASET — PlantVillage, {config.NUM_CLASSES} classes")
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
        print(f"   {classe:<50} {len(fichiers):>5} images")
    print(f"   Images illisibles écartées : {len(invalides)}")
    print(f"   Formats rencontrés : {dict(tailles.most_common(3))}")

    # --- Doublons (global) ------------------------------------------------
    print("\n2. Doublons (analyse globale, toutes classes confondues)")
    analyse = detecter_doublons(images, verifier_quasi=not arguments.sans_quasi_doublons)
    for cle, valeur in analyse.statistiques.items():
        print(f"   {cle:<34} {valeur}")
    if analyse.conflits:
        print("   ATTENTION : images identiques étiquetées dans plusieurs classes :")
        for conflit in analyse.conflits[:10]:
            print(f"     {conflit['classes']} → {conflit['fichier'] if 'fichier' in conflit else conflit['fichiers']}")

    chemin_rapport = config.METRICS_DIR / "qualite_donnees.json"
    sauvegarder_json(analyse.rapport(), chemin_rapport)
    print(f"   Rapport détaillé : {chemin_rapport}")

    if arguments.analyse_doublons:
        print("\nAnalyse terminée (aucun découpage écrit).")
        return 0

    # --- Découpage --------------------------------------------------------
    decoupage, details = construire_decoupage(
        analyse.images,
        ratio_val=arguments.val_size,
        ratio_test=arguments.test_size,
        graine=arguments.seed,
        groupes=analyse.groupes_intra,
    )

    if arguments.force:
        for split in SPLITS:
            dossier_split = arguments.output_dir / split
            if dossier_split.exists():
                shutil.rmtree(dossier_split)
            dossier_split.mkdir(parents=True, exist_ok=True)
    ecrire_ensembles(decoupage, arguments.output_dir, arguments.copy)

    # --- Manifeste --------------------------------------------------------
    comptages = {
        split: {classe: len(decoupage[split][classe]) for classe in config.CLASS_NAMES}
        for split in SPLITS
    }
    totaux = {split: sum(comptages[split].values()) for split in SPLITS}
    total_general = sum(totaux.values())

    manifeste = {
        "dataset": (
            f"PlantVillage (images couleur, {config.NUM_CLASSES} classes, "
            f"{len({classe.split('___')[0] for classe in config.CLASS_NAMES})} cultures)"
        ),
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
        "classes": config.CLASS_NAMES,
        "comptages": comptages,
        "totaux": totaux,
        "total_general": total_general,
        "images_brutes": {classe: len(fichiers) for classe, fichiers in images.items()},
        "doublons": {
            "methode": "MD5 global avant découpage",
            "groupes_exacts": analyse.statistiques["groupes_doublons_exacts"],
            "conflits_annotation": analyse.statistiques["conflits_annotation"],
            "images_ecartees_conflits": analyse.statistiques["images_ecartees_conflits"],
            "groupes_quasi_doublons": analyse.statistiques.get("groupes_quasi_doublons", 0),
            "rapport_detaille": str(chemin_rapport.relative_to(config.BASE_DIR)),
        },
        "images_utilisees": details["images_utilisees"],
        "proportions_obtenues": {
            split: round(totaux[split] / total_general, 4) for split in SPLITS
        },
        "classes_rares_ecartees_des_ratios": details[
            "classes_rares_ecartees_des_ratios"
        ],
        "images_invalides": invalides,
        "mise_en_lien": "copie" if arguments.copy else "lien physique (hardlink)",
    }
    sauvegarder_json(manifeste, manifeste_existant)

    # --- Rapport console --------------------------------------------------
    print("\n3. Découpage stratifié")
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

    print("\n4. Proportions réellement obtenues")
    for split in SPLITS:
        print(
            f"   {split:<11}{totaux[split]:>8} images"
            f"{100 * totaux[split] / total_general:>8.2f} %"
        )
    rares = details["classes_rares_ecartees_des_ratios"]
    print(
        f"   Classes où le minimum de {config.MIN_IMAGES_PER_SPLIT} images "
        f"par ensemble domine : {len(rares)}"
    )
    for rare in rares[:5]:
        print(
            f"     {rare['classe']:<45}{rare['images']:>6} images -> "
            f"{100 * rare['part_train']:.0f}/{100 * rare['part_validation']:.0f}/"
            f"{100 * rare['part_test']:.0f} %"
        )

    print("\n5. Contrôles")
    print(f"   Graine aléatoire          : {arguments.seed}")
    print(
        f"   Doublons exacts           : {analyse.statistiques['groupes_doublons_exacts']} groupes "
        "(regroupés dans le même ensemble)"
    )
    print(f"   Conflits d'annotation     : {analyse.statistiques['conflits_annotation']}")
    print(f"   Fichiers illisibles       : {len(invalides)}")
    print(f"   Manifeste                 : {manifeste_existant}")
    print("\nPréparation terminée. Étape suivante : python training/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
