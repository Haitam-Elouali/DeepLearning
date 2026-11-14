"""Entraînement du modèle MobileNetV2 par Transfer Learning (deux phases).

Phase 1 — Extraction de caractéristiques :
    la base MobileNetV2 (pré-entraînée sur ImageNet) est **gelée** ; seule la
    nouvelle tête de classification (GAP → Dense → Softmax) est entraînée.

Phase 2 — Fine-Tuning partiel :
    les dernières couches de la base sont **dégelées** et ré-entraînées avec un
    learning rate très faible, afin d'adapter les filtres les plus spécifiques
    au domaine des feuilles de tomate.

Le meilleur modèle (au sens de l'accuracy de validation) est sauvegardé dans
``models/best_model.keras``, avec la configuration complète de l'expérience.

Usage :
    python training/train.py
    python training/train.py --epochs 15 --fine-tune-epochs 5 --batch-size 32
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # logs TensorFlow concis

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
from tensorflow.keras.applications import MobileNetV2  # noqa: E402
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # noqa: E402

import config  # noqa: E402
from utils.helpers import sauvegarder_json  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
def fixer_graines(graine: int) -> None:
    """Fixe toutes les graines aléatoires disponibles."""
    os.environ["PYTHONHASHSEED"] = str(graine)
    keras.utils.set_random_seed(graine)


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
def construire_augmentation() -> keras.Sequential:
    """Data Augmentation appliquée uniquement à l'ensemble d'entraînement.

    Les couches sont inactives à l'inférence et ne font pas partie du modèle
    sauvegardé : le prétraitement de prédiction reste donc simplement
    « redimensionnement + normalisation ».
    """
    return keras.Sequential(
        [
            layers.RandomRotation(factor=0.05, fill_mode="nearest"),
            layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="nearest"),
            layers.RandomFlip(mode="horizontal"),
            layers.RandomBrightness(factor=0.15, value_range=(0, 255)),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="nearest"),
        ],
        name="augmentation",
    )


def charger_ensembles(
    dossier_donnees: Path, batch_size: int, graine: int
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Charge les trois ensembles et construit les pipelines tf.data."""
    parametres_communs = dict(
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        batch_size=batch_size,
    )

    train_ds = keras.utils.image_dataset_from_directory(
        dossier_donnees / "train", shuffle=True, seed=graine, **parametres_communs
    )
    val_ds = keras.utils.image_dataset_from_directory(
        dossier_donnees / "validation", shuffle=False, **parametres_communs
    )
    test_ds = keras.utils.image_dataset_from_directory(
        dossier_donnees / "test", shuffle=False, **parametres_communs
    )

    augmentation = construire_augmentation()

    def augmenter(images, etiquettes):
        return augmentation(images, training=True), etiquettes

    def normaliser(images, etiquettes):
        # preprocess_input de MobileNetV2 : [0, 255] -> [-1, 1]
        return preprocess_input(images), etiquettes

    train_ds = (
        train_ds.map(augmenter, num_parallel_calls=AUTOTUNE)
        .map(normaliser, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.map(normaliser, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(normaliser, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


def poids_des_classes(dossier_train: Path) -> dict[int, float]:
    """Calcule des poids équilibrés (compense le déséquilibre entre classes)."""
    comptages = np.array(
        [
            len([f for f in (dossier_train / classe).iterdir() if f.is_file()])
            for classe in config.CLASS_NAMES
        ],
        dtype=np.float64,
    )
    total = comptages.sum()
    poids = total / (len(comptages) * np.maximum(comptages, 1))
    return {index: float(poids[index]) for index in range(len(config.CLASS_NAMES))}


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
def construire_modele(nombre_classes: int, learning_rate: float) -> tuple[keras.Model, keras.Model]:
    """Construit MobileNetV2 + tête de classification.

    Returns
    -------
    tuple
        ``(modele, base)`` — le modèle complet et la base pré-entraînée (dont
        les couches seront partiellement dégelées en phase 2).
    """
    base = MobileNetV2(
        input_shape=(*config.IMAGE_SIZE, 3),
        include_top=False,  # on supprime la couche de classification ImageNet
        weights="imagenet",
    )
    base.trainable = False  # phase 1 : base entièrement gelée

    entree = layers.Input(shape=(*config.IMAGE_SIZE, 3), name="image")
    x = base(entree, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_caracteristiques")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    sortie = layers.Dense(nombre_classes, activation="softmax", name="probabilites")(x)

    modele = keras.Model(entree, sortie, name="mobilenetv2_tomate")
    modele.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modele, base


def degeler_fines_couches(base: keras.Model, nombre_couches: int) -> int:
    """Dégèle les ``nombre_couches`` dernières couches de la base.

    Les couches BatchNormalization restent gelées : avec de petits lots, leur
    mise à jour déstabilise l'apprentissage.
    """
    base.trainable = True
    limite = len(base.layers) - nombre_couches
    for index, couche in enumerate(base.layers):
        couche.trainable = index >= limite and not isinstance(
            couche, layers.BatchNormalization
        )
    return sum(1 for couche in base.layers if couche.trainable)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def construire_callbacks(
    chemin_modele: Path,
    chemin_csv: Path,
    patience: int,
    seuil_initial: float | None = None,
) -> list[keras.callbacks.Callback]:
    """Early Stopping, Model Checkpoint, réduction du LR et journal CSV."""
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(chemin_modele),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            initial_value_threshold=seuil_initial,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(chemin_csv), separator=",", append=True),
    ]
    return callbacks


def fusionner_historiques(historiques: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    """Concatène les historiques des deux phases pour les courbes finales."""
    fusion: dict[str, list[float]] = {}
    for historique in historiques:
        for cle, valeurs in historique.items():
            fusion.setdefault(cle, []).extend([float(v) for v in valeurs])
    return fusion


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main() -> int:
    parseur = argparse.ArgumentParser(description="Entraînement MobileNetV2 (tomate).")
    parseur.add_argument("--data-dir", type=Path, default=config.PROCESSED_DATA_DIR)
    parseur.add_argument("--epochs", type=int, default=config.EPOCHS)
    parseur.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    parseur.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parseur.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parseur.add_argument(
        "--fine-tune-learning-rate", type=float, default=config.FINE_TUNE_LEARNING_RATE
    )
    parseur.add_argument("--fine-tune-layers", type=int, default=config.FINE_TUNE_LAYERS)
    parseur.add_argument("--no-fine-tune", action="store_true", help="Désactive la phase 2.")
    parseur.add_argument("--no-class-weights", action="store_true")
    parseur.add_argument("--seed", type=int, default=config.SEED)
    arguments = parseur.parse_args()

    fixer_graines(arguments.seed)

    if not (arguments.data_dir / "train").is_dir():
        raise SystemExit(
            "Données préparées introuvables. Lancez d'abord : python training/prepare_data.py"
        )

    print("=" * 72)
    print(" ENTRAÎNEMENT — MobileNetV2 + Transfer Learning (4 classes de tomate)")
    print("=" * 72)
    print(f" TensorFlow : {tf.__version__} | Python : {platform.python_version()}")
    print(f" Images     : {config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]} | batch : {arguments.batch_size}")
    print(f" Graine     : {arguments.seed}")

    train_ds, val_ds, test_ds = charger_ensembles(
        arguments.data_dir, arguments.batch_size, arguments.seed
    )

    comptages = {
        split: {
            classe: len([f for f in (arguments.data_dir / split / classe).iterdir() if f.is_file()])
            for classe in config.CLASS_NAMES
        }
        for split in ("train", "validation", "test")
    }
    print(f" Train : {sum(comptages['train'].values())} images | "
          f"Validation : {sum(comptages['validation'].values())} | "
          f"Test : {sum(comptages['test'].values())} (non utilisé ici)")

    modele, base = construire_modele(config.NUM_CLASSES, arguments.learning_rate)

    resume_lignes: list[str] = []
    modele.summary(print_fn=lambda ligne: resume_lignes.append(ligne), expand_nested=False)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR / "model_summary.txt").write_text(
        "\n".join(resume_lignes), encoding="utf-8"
    )
    entrainables = int(sum(np.prod(p.shape) for p in modele.trainable_weights))
    total_parametres = int(modele.count_params())
    print(f" Paramètres : {total_parametres:,} au total, {entrainables:,} entraînables (phase 1)")

    class_weight = None if arguments.no_class_weights else poids_des_classes(
        arguments.data_dir / "train"
    )
    if class_weight:
        print(
            " Poids de classes : "
            + ", ".join(
                f"{config.CLASS_NAMES[i].split('___')[-1]}={p:.2f}"
                for i, p in class_weight.items()
            )
        )

    chemin_csv = config.METRICS_DIR / "training_log.csv"
    chemin_csv.unlink(missing_ok=True)
    chemin_phase1 = config.MODELS_DIR / "phase1_features.keras"

    # ------------------------------------------------------------------
    # Phase 1 : extraction de caractéristiques
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print(" PHASE 1 — Extraction de caractéristiques (base gelée)")
    print("-" * 72)
    debut_phase1 = time.perf_counter()
    historique_1 = modele.fit(
        train_ds,
        validation_data=val_ds,
        epochs=arguments.epochs,
        class_weight=class_weight,
        callbacks=construire_callbacks(
            chemin_phase1, chemin_csv, config.EARLY_STOPPING_PATIENCE
        ),
        verbose=2,
    )
    duree_phase1 = time.perf_counter() - debut_phase1
    meilleure_val_1 = float(max(historique_1.history.get("val_accuracy", [0.0])))
    print(f" Phase 1 terminée en {duree_phase1 / 60:.1f} min — "
          f"meilleure accuracy de validation : {meilleure_val_1:.4f}")

    # Le meilleur modèle de la phase 1 devient la référence.
    modele = keras.models.load_model(chemin_phase1)
    modele.save(config.MODEL_PATH)

    historiques = [historique_1.history]
    parametres_phase2: dict[str, object] = {"active": not arguments.no_fine_tune}

    # ------------------------------------------------------------------
    # Phase 2 : Fine-Tuning partiel
    # ------------------------------------------------------------------
    if not arguments.no_fine_tune and arguments.fine_tune_epochs > 0:
        print("\n" + "-" * 72)
        print(f" PHASE 2 — Fine-Tuning des {arguments.fine_tune_layers} dernières couches")
        print("-" * 72)

        # On récupère la base MobileNetV2 à l'intérieur du modèle rechargé.
        base_rechargee = next(
            couche
            for couche in modele.layers
            if isinstance(couche, keras.Model) and couche.name.startswith("mobilenetv2")
        )
        nombre_degeles = degeler_fines_couches(base_rechargee, arguments.fine_tune_layers)
        modele.compile(
            optimizer=keras.optimizers.Adam(learning_rate=arguments.fine_tune_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(f" Couches dégelées : {nombre_degeles}")

        debut_phase2 = time.perf_counter()
        historique_2 = modele.fit(
            train_ds,
            validation_data=val_ds,
            epochs=len(historique_1.history["loss"]) + arguments.fine_tune_epochs,
            initial_epoch=len(historique_1.history["loss"]),
            class_weight=class_weight,
            callbacks=construire_callbacks(
                config.MODEL_PATH,
                chemin_csv,
                config.EARLY_STOPPING_PATIENCE,
                seuil_initial=meilleure_val_1,
            ),
            verbose=2,
        )
        duree_phase2 = time.perf_counter() - debut_phase2
        historiques.append(historique_2.history)
        parametres_phase2.update(
            {
                "couches_degelees": nombre_degeles,
                "epochs": arguments.fine_tune_epochs,
                "learning_rate": arguments.fine_tune_learning_rate,
                "duree_min": round(duree_phase2 / 60, 2),
            }
        )
        print(f" Phase 2 terminée en {duree_phase2 / 60:.1f} min")
    else:
        duree_phase2 = 0.0

    # ------------------------------------------------------------------
    # Artefacts
    # ------------------------------------------------------------------
    historique_fusionne = fusionner_historiques(historiques)
    sauvegarder_json(historique_fusionne, config.HISTORY_PATH)

    meilleure_val = float(max(historique_fusionne.get("val_accuracy", [0.0])))
    meilleure_epoch = int(np.argmax(historique_fusionne.get("val_accuracy", [0.0]))) + 1

    configuration = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "modele": "MobileNetV2 (ImageNet) + GlobalAveragePooling2D + Dense(128) + Dense(4, softmax)",
        "tensorflow": tf.__version__,
        "python": platform.python_version(),
        "graine": arguments.seed,
        "taille_image": list(config.IMAGE_SIZE),
        "batch_size": arguments.batch_size,
        "optimiseur": "Adam",
        "perte": "categorical_crossentropy",
        "phase1": {
            "epochs_max": arguments.epochs,
            "epochs_effectues": len(historique_1.history["loss"]),
            "learning_rate": arguments.learning_rate,
            "base_gelee": True,
            "meilleure_val_accuracy": meilleure_val_1,
            "duree_min": round(duree_phase1 / 60, 2),
        },
        "phase2_fine_tuning": parametres_phase2,
        "callbacks": {
            "early_stopping": {"monitor": "val_loss", "patience": config.EARLY_STOPPING_PATIENCE},
            "model_checkpoint": {"monitor": "val_accuracy", "save_best_only": True},
            "reduce_lr": {
                "monitor": "val_loss",
                "facteur": config.REDUCE_LR_FACTOR,
                "patience": config.REDUCE_LR_PATIENCE,
            },
        },
        "augmentation": [
            "rotation ±5 %",
            "zoom ±10 %",
            "retournement horizontal",
            "luminosité ±15 %",
            "translation ±5 %",
        ],
        "poids_de_classes": class_weight,
        "comptages": comptages,
        "parametres_totaux": total_parametres,
        "parametres_entrainables_phase1": entrainables,
        "meilleure_val_accuracy": meilleure_val,
        "meilleure_epoch": meilleure_epoch,
        "epochs_total": len(historique_fusionne.get("loss", [])),
        "classes": config.CLASS_NAMES,
        "fichier_modele": config.MODEL_PATH.name,
    }
    sauvegarder_json(configuration, config.TRAINING_CONFIG_PATH)
    sauvegarder_json(config.CLASS_NAMES, config.CLASS_NAMES_PATH)

    print("\n" + "=" * 72)
    print(f" Meilleure accuracy de validation : {meilleure_val:.4f} (époque {meilleure_epoch})")
    print(f" Modèle sauvegardé : {config.MODEL_PATH}")
    print(f" Configuration     : {config.TRAINING_CONFIG_PATH}")
    print(f" Historique        : {config.HISTORY_PATH}")
    print(" Étape suivante    : python training/evaluate.py")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
