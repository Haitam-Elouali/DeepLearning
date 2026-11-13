"""Chargement unique du modèle Keras et réalisation des prédictions.

Le modèle est coûteux à charger : il est donc instancié **une seule fois** au
démarrage de l'application (singleton protégé par un verrou), puis réutilisé
pour toutes les requêtes.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import config
from utils.preprocessing import preparer_image, preparer_lot, preparer_pour_modele

__all__ = ["PredictionError", "Predictor", "obtenir_predictor", "modele_disponible"]


class PredictionError(RuntimeError):
    """Erreur fonctionnelle levée lorsqu'une prédiction est impossible."""


class Predictor:
    """Encapsule le modèle entraîné et la liste ordonnée des classes."""

    def __init__(
        self,
        chemin_modele: Path | None = None,
        chemin_classes: Path | None = None,
    ) -> None:
        self.chemin_modele = Path(chemin_modele or config.MODEL_PATH)
        self.chemin_classes = Path(chemin_classes or config.CLASS_NAMES_PATH)
        self._modele: Any = None
        self._classes: list[str] | None = None
        self._verrou = threading.Lock()
        self._version: str = ""

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    @property
    def charge(self) -> bool:
        """Indique si le modèle est déjà en mémoire."""
        return self._modele is not None

    @property
    def version(self) -> str:
        """Nom du fichier modèle (affiché dans l'interface, sans chemin absolu)."""
        return self._version or self.chemin_modele.name

    def charger(self) -> None:
        """Charge le modèle et les classes (idempotent, thread-safe)."""
        if self._modele is not None:
            return
        with self._verrou:
            if self._modele is not None:  # double vérification
                return
            if not self.chemin_modele.exists():
                raise PredictionError(
                    "Aucun modèle entraîné n'est disponible. "
                    "Exécutez d'abord « python training/train.py »."
                )
            if not self.chemin_classes.exists():
                raise PredictionError(
                    "Le fichier des classes (data/class_names.json) est absent. "
                    "Relancez la préparation des données puis l'entraînement."
                )

            # Import différé : l'application peut démarrer (et les tests
            # unitaires s'exécuter) même si TensorFlow n'est pas installé.
            from tensorflow import keras

            try:
                self._modele = keras.models.load_model(self.chemin_modele)
            except Exception as exc:  # noqa: BLE001 - message clair pour l'utilisateur
                raise PredictionError(
                    "Le modèle n'a pas pu être chargé. Vérifiez qu'il a été "
                    "entraîné avec la même version de TensorFlow/Keras."
                ) from exc

            with self.chemin_classes.open("r", encoding="utf-8") as fichier:
                classes = json.load(fichier)
            if isinstance(classes, dict):  # tolérance : format {classe: index}
                classes = [nom for nom, _ in sorted(classes.items(), key=lambda kv: kv[1])]
            self._classes = list(classes)
            self._version = self.chemin_modele.name

    @property
    def classes(self) -> list[str]:
        """Liste ordonnée des classes (index = position dans la sortie softmax)."""
        if self._classes is None:
            self.charger()
        return list(self._classes or [])

    @property
    def modele(self) -> Any:
        if self._modele is None:
            self.charger()
        return self._modele

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------
    def _probabilites_vers_dict(self, vecteur: np.ndarray) -> dict[str, float]:
        classes = self.classes
        if len(vecteur) != len(classes):
            raise PredictionError(
                "Incohérence entre la sortie du modèle et les classes connues."
            )
        return {classe: float(score) for classe, score in zip(classes, vecteur)}

    def predire(self, source: str | Path | Image.Image) -> dict[str, Any]:
        """Prédit la classe d'une image (chemin de fichier ou image Pillow).

        Returns
        -------
        dict
            ``classe``, ``indice``, ``confiance``, ``probabilites`` (dict),
            ``classement`` (liste triée), ``duree_ms``.
        """
        tenseur = (
            preparer_image(source)
            if isinstance(source, Image.Image)
            else preparer_pour_modele(source)
        )
        lot = np.expand_dims(tenseur, axis=0)

        debut = time.perf_counter()
        try:
            sortie = self.modele.predict(lot, verbose=0)
        except Exception as exc:  # noqa: BLE001
            raise PredictionError("L'analyse de l'image a échoué.") from exc
        duree_ms = int((time.perf_counter() - debut) * 1000)

        vecteur = np.asarray(sortie, dtype=np.float64).reshape(-1)
        total = float(vecteur.sum())
        if total > 0:
            vecteur = vecteur / total  # sécurité numérique (softmax déjà normalisé)

        probabilites = self._probabilites_vers_dict(vecteur)
        indice = int(np.argmax(vecteur))
        classe = self.classes[indice]
        classement = sorted(
            (
                {"classe": nom, "probabilite": score}
                for nom, score in probabilites.items()
            ),
            key=lambda element: element["probabilite"],
            reverse=True,
        )

        return {
            "classe": classe,
            "indice": indice,
            "confiance": float(vecteur[indice]),
            "probabilites": probabilites,
            "classement": classement,
            "duree_ms": duree_ms,
        }

    def predire_lot(self, chemins: list[str | Path]) -> np.ndarray:
        """Prédit un lot d'images (utilisé par l'évaluation et l'analyse d'erreurs)."""
        if not chemins:
            return np.empty((0, len(self.classes)), dtype=np.float32)
        lot = preparer_lot(chemins)
        return np.asarray(self.modele.predict(lot, batch_size=config.BATCH_SIZE, verbose=0))


# ---------------------------------------------------------------------------
# Singleton applicatif
# ---------------------------------------------------------------------------
_predictor: Predictor | None = None
_verrou_predictor = threading.Lock()


def obtenir_predictor() -> Predictor:
    """Retourne l'instance unique du prédicteur (créée au premier appel)."""
    global _predictor
    if _predictor is None:
        with _verrou_predictor:
            if _predictor is None:
                _predictor = Predictor()
    return _predictor


def modele_disponible() -> bool:
    """Indique si le fichier modèle et le fichier des classes existent."""
    return config.MODEL_PATH.exists() and config.CLASS_NAMES_PATH.exists()
