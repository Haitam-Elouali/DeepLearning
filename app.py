"""Application Web minimale — maladie de la feuille de tomate.

Lancement :

    python app.py

puis ouvrir http://127.0.0.1:5000

L'application ne fait qu'une chose : recevoir une image, la passer au modèle
entraîné et afficher les quatre probabilités. Aucune image n'est enregistrée.
"""

from __future__ import annotations

import logging

from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

import config
from utils.helpers import (
    classe_est_saine,
    formater_pourcentage,
    nom_affichage,
    valider_image,
)
from utils.predictor import PredictionError, obtenir_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
)
logger = logging.getLogger("tomatescan")


def creer_application() -> Flask:
    """Construit l'application Flask (fabrique réutilisable par les tests)."""
    application = Flask(__name__)
    application.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE

    predict = obtenir_predictor()
    erreur_modele: str | None = None
    try:
        predict.charger()
        logger.info("Modèle chargé : %s (%d classes)", predict.version, len(predict.classes))
    except PredictionError as exc:
        erreur_modele = str(exc)
        logger.warning("Modèle indisponible : %s", erreur_modele)

    def rendre(**contexte):
        """Rend la page unique avec le contexte commun."""
        return render_template(
            "index.html",
            app_nom=config.APP_NAME,
            classes=config.CLASSES_FR,
            modele_pret=erreur_modele is None,
            erreur_modele=erreur_modele,
            **contexte,
        )

    @application.template_filter("pourcentage")
    def filtre_pourcentage(valeur: float) -> str:
        return formater_pourcentage(valeur)

    @application.get("/")
    def accueil():
        """Affiche le formulaire de téléversement."""
        return rendre()

    @application.post("/analyser")
    def analyser():
        """Reçoit une image, prédit sa classe et affiche le résultat."""
        if erreur_modele is not None:
            return rendre(erreur=erreur_modele), 503

        try:
            image = valider_image(request.files.get("image"))
            resultat = predict.predire(image)
        except ValueError as exc:
            return rendre(erreur=str(exc)), 400
        except PredictionError:
            logger.exception("Échec de la prédiction")
            return rendre(erreur="L'analyse de l'image a échoué. Réessayez."), 500

        classe = resultat["classe"]
        probabilites = [
            {
                "nom_fr": nom_affichage(element["classe"]),
                "pourcentage": formater_pourcentage(element["probabilite"], 1),
            }
            for element in resultat["classement"]
        ]
        return rendre(
            resultat={
                "classe": nom_affichage(classe),
                "saine": classe_est_saine(classe),
                "confiance": resultat["confiance"],
                "duree_ms": resultat["duree_ms"],
            },
            probabilites=probabilites,
        )

    @application.errorhandler(RequestEntityTooLarge)
    def image_trop_volumineuse(_erreur):
        return rendre(
            erreur=f"Image trop volumineuse. Taille maximale : "
            f"{config.MAX_UPLOAD_SIZE // (1024 * 1024)} Mo."
        ), 413

    @application.errorhandler(404)
    def page_introuvable(_erreur):
        return rendre(erreur="Cette page n'existe pas. Revenez à l'accueil."), 404

    return application


app = creer_application()


if __name__ == "__main__":
    logger.info("Serveur démarré sur http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
