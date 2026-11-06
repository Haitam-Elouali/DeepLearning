"""Accès à la base de données SQLite de l'historique des analyses.

Le module encapsule tout le SQL : l'application Flask ne manipule que des
dictionnaires Python. Une connexion est ouverte par opération (SQLite est très
rapide en local et cela évite tout problème d'accès concurrent entre les
threads du serveur Web).

Aucune information sensible n'est stockée : uniquement le nom du fichier
téléversé, la classe prédite, les probabilités et l'horodatage.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import config

__all__ = [
    "initialiser_base",
    "ajouter_analyse",
    "lister_analyses",
    "obtenir_analyse",
    "supprimer_analyse",
    "compter_analyses",
    "statistiques",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nom_stocke      TEXT    NOT NULL,
    nom_original    TEXT    NOT NULL,
    classe_predite  TEXT    NOT NULL,
    confiance       REAL    NOT NULL,
    probabilites    TEXT    NOT NULL,
    modele          TEXT    NOT NULL DEFAULT '',
    duree_ms        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC);
"""


def _connexion(chemin: Path | None = None) -> sqlite3.Connection:
    """Ouvre une connexion configurée (lignes accessibles comme des dicts)."""
    chemin = Path(chemin or config.DATABASE_PATH)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    connexion = sqlite3.connect(chemin, timeout=10)
    connexion.row_factory = sqlite3.Row
    connexion.execute("PRAGMA journal_mode=WAL")
    connexion.execute("PRAGMA foreign_keys=ON")
    return connexion


def initialiser_base(chemin: Path | None = None) -> None:
    """Crée la table et l'index si nécessaire (idempotent)."""
    with closing(_connexion(chemin)) as connexion:
        connexion.executescript(SCHEMA)
        connexion.commit()


def _ligne_vers_dict(ligne: sqlite3.Row) -> dict[str, Any]:
    """Convertit une ligne SQL en dictionnaire prêt pour les templates."""
    analyse = dict(ligne)
    try:
        analyse["probabilites"] = json.loads(analyse.get("probabilites") or "{}")
    except json.JSONDecodeError:
        analyse["probabilites"] = {}
    analyse["confiance_pct"] = round(float(analyse["confiance"]) * 100, 1)
    analyse["date_affichage"] = datetime.fromisoformat(analyse["created_at"]).strftime(
        "%d/%m/%Y à %H:%M"
    )
    return analyse


def ajouter_analyse(
    nom_stocke: str,
    nom_original: str,
    classe_predite: str,
    confiance: float,
    probabilites: dict[str, float],
    modele: str = "",
    duree_ms: int = 0,
    chemin: Path | None = None,
) -> int:
    """Insère une analyse et renvoie son identifiant."""
    horodatage = datetime.now().isoformat(timespec="seconds")
    with closing(_connexion(chemin)) as connexion:
        curseur = connexion.execute(
            """
            INSERT INTO analyses
                (nom_stocke, nom_original, classe_predite, confiance,
                 probabilites, modele, duree_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nom_stocke,
                nom_original,
                classe_predite,
                float(confiance),
                json.dumps(probabilites, ensure_ascii=False),
                modele,
                int(duree_ms),
                horodatage,
            ),
        )
        connexion.commit()
        return int(curseur.lastrowid)


def lister_analyses(limite: int = 50, chemin: Path | None = None) -> list[dict[str, Any]]:
    """Retourne les analyses les plus récentes (ordre décroissant)."""
    with closing(_connexion(chemin)) as connexion:
        lignes = connexion.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (int(limite),)
        ).fetchall()
    return [_ligne_vers_dict(ligne) for ligne in lignes]


def obtenir_analyse(identifiant: int, chemin: Path | None = None) -> dict[str, Any] | None:
    """Retourne une analyse par identifiant, ou ``None`` si absente."""
    with closing(_connexion(chemin)) as connexion:
        ligne = connexion.execute(
            "SELECT * FROM analyses WHERE id = ?", (int(identifiant),)
        ).fetchone()
    return _ligne_vers_dict(ligne) if ligne else None


def supprimer_analyse(identifiant: int, chemin: Path | None = None) -> bool:
    """Supprime une analyse. Renvoie ``True`` si une ligne a été effacée."""
    with closing(_connexion(chemin)) as connexion:
        curseur = connexion.execute("DELETE FROM analyses WHERE id = ?", (int(identifiant),))
        connexion.commit()
        return curseur.rowcount > 0


def compter_analyses(chemin: Path | None = None) -> int:
    """Nombre total d'analyses enregistrées."""
    with closing(_connexion(chemin)) as connexion:
        ligne = connexion.execute("SELECT COUNT(*) AS total FROM analyses").fetchone()
    return int(ligne["total"]) if ligne else 0


def statistiques(chemin: Path | None = None) -> dict[str, Any]:
    """Statistiques agrégées affichées sur le tableau de bord."""
    with closing(_connexion(chemin)) as connexion:
        total = connexion.execute("SELECT COUNT(*) AS total FROM analyses").fetchone()["total"]
        confiance = connexion.execute(
            "SELECT AVG(confiance) AS moyenne FROM analyses"
        ).fetchone()["moyenne"]
        par_classe = {
            ligne["classe_predite"]: ligne["nombre"]
            for ligne in connexion.execute(
                "SELECT classe_predite, COUNT(*) AS nombre FROM analyses GROUP BY classe_predite"
            ).fetchall()
        }
        depuis_7_jours = connexion.execute(
            "SELECT COUNT(*) AS total FROM analyses WHERE created_at >= ?",
            ((datetime.now() - timedelta(days=7)).isoformat(timespec="seconds"),),
        ).fetchone()["total"]
        derniere = connexion.execute(
            "SELECT created_at FROM analyses ORDER BY id DESC LIMIT 1"
        ).fetchone()

    return {
        "total": int(total or 0),
        "confiance_moyenne": float(confiance) if confiance is not None else None,
        "par_classe": par_classe,
        "semaine": int(depuis_7_jours or 0),
        "derniere_analyse": derniere["created_at"] if derniere else None,
    }
