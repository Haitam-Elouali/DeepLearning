"""Application Web Flask — détection des maladies de la feuille de tomate.

Lancement :
    python app.py
    # ou : flask --app app run --debug

Principes :
- le modèle Keras est chargé **une seule fois** au démarrage (singleton) ;
- aucun chemin interne n'est exposé dans les réponses ;
- tous les fichiers téléversés sont validés (extension, type MIME, décodage
  réel par Pillow, dimensions) puis ré-encodés en JPEG et renommés ;
- les erreurs sont journalisées côté serveur et renvoyées de façon générique.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

import config
from utils import database
from utils.helpers import (
    charger_infos_maladies,
    charger_json,
    classe_est_saine,
    enregistrer_upload,
    formater_date,
    formater_pourcentage,
    infos_classe,
    nom_affichage,
)
from utils.predictor import PredictionError, obtenir_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
)
logger = logging.getLogger("tomatescan")

NOM_FICHIER_UPLOAD = re.compile(r"^[0-9a-f]{32}\.jpg$")


def creer_application() -> Flask:
    """Fabrique Flask (utile pour les tests : configuration isolable)."""
    application = Flask(__name__)
    application.config.update(
        SECRET_KEY=config.SECRET_KEY,
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_SIZE,
        JSON_AS_ASCII=False,
        TEMPLATES_AUTO_RELOAD=False,
    )

    database.initialiser_base()

    # ------------------------------------------------------------------
    # Chargement unique du modèle
    # ------------------------------------------------------------------
    predict = obtenir_predictor()
    erreur_modele: str | None = None
    try:
        predict.charger()
        logger.info("Modèle chargé : %s (%d classes)", predict.version, len(predict.classes))
    except PredictionError as exc:
        erreur_modele = str(exc)
        logger.warning("Modèle indisponible : %s", erreur_modele)
    application.config["ERREUR_MODELE"] = erreur_modele
    application.config["PREDICTOR"] = predict

    # ------------------------------------------------------------------
    # Contexte commun aux templates
    # ------------------------------------------------------------------
    @application.context_processor
    def injecter_contexte():  # noqa: ANN202
        return {
            "app_nom": config.APP_NAME,
            "app_titre": config.APP_FULL_NAME,
            "app_version": config.APP_VERSION,
            "annee": 2026,
        }

    @application.template_filter("pourcentage")
    def filtre_pourcentage(valeur):  # noqa: ANN001, ANN202
        return formater_pourcentage(valeur)

    @application.template_filter("date_fr")
    def filtre_date(valeur):  # noqa: ANN001, ANN202
        return formater_date(valeur)

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @application.get("/")
    def accueil():  # noqa: ANN202
        stats = database.statistiques()
        configuration = charger_json(config.TRAINING_CONFIG_PATH, defaut={}) or {}
        manifeste = charger_json(config.DATASET_MANIFEST_PATH, defaut={}) or {}
        analyses_recentes = database.lister_analyses(limite=5)
        for analyse in analyses_recentes:
            analyse["nom_fr"] = nom_affichage(analyse["classe_predite"])
            analyse["saine"] = classe_est_saine(analyse["classe_predite"])
        classes = [
            {
                "cle": classe,
                **infos_classe(classe),
                "analyses": stats["par_classe"].get(classe, 0),
                "exemple": infos_classe(classe).get("image_exemple", ""),
            }
            for classe in config.CLASS_NAMES
        ]
        return render_template(
            "index.html",
            page="accueil",
            stats=stats,
            classes=classes,
            analyses_recentes=analyses_recentes,
            configuration=configuration,
            manifeste=manifeste,
            modele_pret=erreur_modele is None,
            erreur_modele=erreur_modele,
        )

    @application.get("/analyse")
    def page_analyse():  # noqa: ANN202
        return render_template(
            "analyze.html",
            page="analyse",
            classes=[{"cle": classe, **infos_classe(classe)} for classe in config.CLASS_NAMES],
            taille_max_mo=config.MAX_UPLOAD_SIZE // (1024 * 1024),
            extensions=sorted(config.ALLOWED_EXTENSIONS),
            modele_pret=erreur_modele is None,
            erreur_modele=erreur_modele,
        )

    @application.get("/historique")
    def page_historique():  # noqa: ANN202
        analyses = database.lister_analyses(limite=config.HISTORY_PAGE_SIZE)
        for analyse in analyses:
            analyse["nom_fr"] = nom_affichage(analyse["classe_predite"])
            analyse["saine"] = classe_est_saine(analyse["classe_predite"])
        return render_template(
            "history.html",
            page="historique",
            analyses=analyses,
            total=database.compter_analyses(),
        )

    @application.get("/analyse/<int:identifiant>")
    def page_detail(identifiant: int):  # noqa: ANN202
        analyse = database.obtenir_analyse(identifiant)
        if analyse is None:
            abort(404)
        infos = infos_classe(analyse["classe_predite"])
        probabilites = [
            {
                "classe": classe,
                "nom_fr": infos_classe(classe)["nom_fr"],
                "valeur": score,
                "pourcentage": round(score * 100, 2),
                "couleur": infos_classe(classe)["couleur"],
            }
            for classe, score in sorted(
                analyse["probabilites"].items(), key=lambda element: element[1], reverse=True
            )
        ]
        return render_template(
            "analysis_detail.html",
            page="historique",
            analyse=analyse,
            infos=infos,
            probabilites=probabilites,
            nom_fr=infos["nom_fr"],
            saine=classe_est_saine(analyse["classe_predite"]),
        )

    @application.get("/maladies")
    def page_maladies():  # noqa: ANN202
        infos = charger_infos_maladies()
        classes = [
            {"cle": classe, **infos.get(classe, infos_classe(classe))}
            for classe in config.CLASS_NAMES
        ]
        manifeste = charger_json(config.DATASET_MANIFEST_PATH, defaut={}) or {}
        return render_template(
            "diseases.html",
            page="maladies",
            classes=classes,
            manifeste=manifeste,
        )

    @application.get("/a-propos")
    def page_a_propos():  # noqa: ANN202
        configuration = charger_json(config.TRAINING_CONFIG_PATH, defaut={}) or {}
        manifeste = charger_json(config.DATASET_MANIFEST_PATH, defaut={}) or {}
        metriques = charger_json(config.METRICS_DIR / "evaluation.json", defaut={}) or {}
        return render_template(
            "about.html",
            page="apropos",
            configuration=configuration,
            manifeste=manifeste,
            metriques=metriques,
        )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    @application.post("/api/analyser")
    def api_analyser():  # noqa: ANN202
        """Reçoit une image, prédit la classe et enregistre l'analyse."""
        if erreur_modele is not None:
            return (
                jsonify(
                    {
                        "succes": False,
                        "erreur": "Le modèle de prédiction n'est pas disponible. "
                        "Contactez l'administrateur de l'application.",
                    }
                ),
                503,
            )

        fichier = request.files.get("image")
        if fichier is None:
            return jsonify({"succes": False, "erreur": "Aucune image reçue."}), 400

        try:
            upload = enregistrer_upload(fichier)
        except ValueError as exc:
            return jsonify({"succes": False, "erreur": str(exc)}), 400
        except OSError:
            logger.exception("Échec de l'enregistrement du fichier")
            return jsonify({"succes": False, "erreur": "Impossible d'enregistrer l'image."}), 500

        try:
            resultat = predict.predire(upload["chemin"])
        except PredictionError:
            logger.exception("Échec de la prédiction")
            upload["chemin"].unlink(missing_ok=True)
            return (
                jsonify(
                    {
                        "succes": False,
                        "erreur": "L'analyse de l'image a échoué. Réessayez avec une autre image.",
                    }
                ),
                500,
            )

        classe = resultat["classe"]
        infos = infos_classe(classe)
        identifiant = database.ajouter_analyse(
            nom_stocke=upload["nom_stocke"],
            nom_original=upload["nom_original"],
            classe_predite=classe,
            confiance=resultat["confiance"],
            probabilites=resultat["probabilites"],
            modele=predict.version,
            duree_ms=resultat["duree_ms"],
        )

        probabilites = [
            {
                "classe": element["classe"],
                "nom_fr": infos_classe(element["classe"])["nom_fr"],
                "valeur": element["probabilite"],
                "pourcentage": round(element["probabilite"] * 100, 2),
                "couleur": infos_classe(element["classe"])["couleur"],
            }
            for element in resultat["classement"]
        ]

        return jsonify(
            {
                "succes": True,
                "analyse": {
                    "id": identifiant,
                    "classe": classe,
                    "classe_nom": infos["nom_fr"],
                    "categorie": infos["categorie"],
                    "niveau": infos["niveau"],
                    "couleur": infos["couleur"],
                    "saine": classe_est_saine(classe),
                    "confiance": round(resultat["confiance"], 4),
                    "confiance_pct": round(resultat["confiance"] * 100, 1),
                    "resume": infos["resume"],
                    "description": infos["description"],
                    "symptomes": infos["symptomes"],
                    "confusion": infos["confusion_possible"],
                    "duree_ms": resultat["duree_ms"],
                    "image_url": url_for("image_televersee", nom_fichier=upload["nom_stocke"]),
                    "nom_image": upload["nom_original"],
                    "detail_url": url_for("page_detail", identifiant=identifiant),
                },
                "probabilites": probabilites,
            }
        )

    @application.delete("/api/historique/<int:identifiant>")
    def api_supprimer(identifiant: int):  # noqa: ANN202
        """Supprime une analyse et son image associée."""
        analyse = database.obtenir_analyse(identifiant)
        if analyse is None:
            return jsonify({"succes": False, "erreur": "Analyse introuvable."}), 404

        nom_stocke = analyse.get("nom_stocke") or ""
        if NOM_FICHIER_UPLOAD.match(nom_stocke):
            (config.UPLOAD_DIR / nom_stocke).unlink(missing_ok=True)

        database.supprimer_analyse(identifiant)
        logger.info("Analyse %s supprimée", identifiant)
        return jsonify({"succes": True, "message": "Analyse supprimée."})

    @application.get("/api/historique")
    def api_historique():  # noqa: ANN202
        """Historique au format JSON (utilisé par le tableau de bord)."""
        analyses = database.lister_analyses(limite=config.HISTORY_PAGE_SIZE)
        for analyse in analyses:
            analyse["nom_fr"] = nom_affichage(analyse["classe_predite"])
        return jsonify({"succes": True, "total": len(analyses), "analyses": analyses})

    @application.get("/api/statistiques")
    def api_statistiques():  # noqa: ANN202
        stats = database.statistiques()
        stats["classes"] = [
            {"classe": classe, "nom_fr": nom_affichage(classe)} for classe in config.CLASS_NAMES
        ]
        return jsonify({"succes": True, "statistiques": stats})

    @application.get("/sante")
    def sante():  # noqa: ANN202
        """Sonde de disponibilité (utile pour les tests et le déploiement)."""
        return jsonify(
            {
                "statut": "ok" if erreur_modele is None else "degradé",
                "modele_charge": predict.charge,
                "classes": len(predict.classes) if predict.charge else 0,
                "version": config.APP_VERSION,
            }
        )

    # ------------------------------------------------------------------
    # Fichiers téléversés
    # ------------------------------------------------------------------
    @application.get("/image/<nom_fichier>")
    def image_televersee(nom_fichier: str):  # noqa: ANN202
        """Sert une image téléversée après validation stricte du nom."""
        if not NOM_FICHIER_UPLOAD.match(nom_fichier):
            abort(404)
        return send_from_directory(config.UPLOAD_DIR, nom_fichier, max_age=3600)

    # ------------------------------------------------------------------
    # Gestion des erreurs
    # ------------------------------------------------------------------
    @application.errorhandler(404)
    def erreur_404(_erreur):  # noqa: ANN001, ANN202
        return (
            render_template(
                "error.html",
                page="erreur",
                code=404,
                titre="Page introuvable",
                message="La page ou l'analyse demandée n'existe pas ou a été supprimée.",
            ),
            404,
        )

    @application.errorhandler(RequestEntityTooLarge)
    def erreur_413(_erreur):  # noqa: ANN001, ANN202
        return (
            jsonify(
                {
                    "succes": False,
                    "erreur": f"Image trop volumineuse. Taille maximale : "
                    f"{config.MAX_UPLOAD_SIZE // (1024 * 1024)} Mo.",
                }
            ),
            413,
        )

    @application.errorhandler(500)
    def erreur_500(_erreur):  # noqa: ANN001, ANN202
        return (
            render_template(
                "error.html",
                page="erreur",
                code=500,
                titre="Erreur interne",
                message="Une erreur inattendue est survenue. L'incident a été enregistré.",
            ),
            500,
        )

    return application


app = creer_application()


if __name__ == "__main__":
    import time as _time

    logger.info("=== DÉMARRAGE DE L'APPLICATION ===")
    t0 = _time.perf_counter()
    logger.info("création de l'application Flask…")
    application = creer_application()
    logger.info("application Flask créée en %.2fs", _time.perf_counter() - t0)

    erreur_modele = application.config.get("ERREUR_MODELE")
    if erreur_modele:
        logger.warning("modèle indisponible : %s", erreur_modele)
    else:
        logger.info("modèle disponible (predictor chargé)")

    logger.info(
        "démarrage du serveur sur http://127.0.0.1:5000 (debug=%s)…",
        False,
    )
    application.run(host="127.0.0.1", port=5000, debug=False)
