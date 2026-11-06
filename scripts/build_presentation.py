"""Génère la présentation de soutenance (PowerPoint, 16:9) à partir des résultats.

Le diaporama est construit automatiquement avec les figures et métriques réelles
du projet, et chaque diapositive contient des notes de présentation (durée
indicative totale : environ 15 minutes).

Usage :
    python scripts/build_presentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

import config  # noqa: E402
from utils.helpers import charger_json, nom_affichage  # noqa: E402

SORTIE = RACINE / "docs" / "presentation" / "presentation_soutenance.pptx"

VERT_FONCE = RGBColor(0x14, 0x53, 0x2D)
VERT = RGBColor(0x15, 0x80, 0x3D)
VERT_CLAIR = RGBColor(0x22, 0xC5, 0x5E)
ROUGE = RGBColor(0xB9, 0x1C, 0x1C)
GRIS = RGBColor(0x4B, 0x55, 0x63)
GRIS_CLAIR = RGBColor(0xF1, 0xF5, 0xE9)
BLANC = RGBColor(0xFF, 0xFF, 0xFF)

LARGEUR = Inches(13.333)
HAUTEUR = Inches(7.5)


def pourcent(valeur: float, decimales: int = 2) -> str:
    return f"{valeur * 100:.{decimales}f} %".replace(".", ",")


class Diaporama:
    """Petite surcouche à python-pptx pour construire un diaporama cohérent."""

    def __init__(self) -> None:
        self.presentation = Presentation()
        self.presentation.slide_width = LARGEUR
        self.presentation.slide_height = HAUTEUR
        self.modele = self.presentation.slide_layouts[6]  # page blanche
        self.numero = 0

    # ------------------------------------------------------------------
    # Éléments de base
    # ------------------------------------------------------------------
    def _zone(self, slide, gauche, haut, largeur, hauteur, couleur=None):
        from pptx.enum.shapes import MSO_SHAPE

        forme = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, gauche, haut, largeur, hauteur)
        if couleur is None:
            forme.fill.background()
        else:
            forme.fill.solid()
            forme.fill.fore_color.rgb = couleur
        forme.line.fill.background()
        return forme

    def _texte(self, slide, gauche, haut, largeur, hauteur, texte, taille=18,
               couleur=GRIS, gras=False, alignement=PP_ALIGN.LEFT, italique=False):
        zone = slide.shapes.add_textbox(gauche, haut, largeur, hauteur)
        cadre = zone.text_frame
        cadre.word_wrap = True
        paragraphe = cadre.paragraphs[0]
        paragraphe.alignment = alignement
        run = paragraphe.add_run()
        run.text = texte
        run.font.size = Pt(taille)
        run.font.bold = gras
        run.font.italic = italique
        run.font.color.rgb = couleur
        run.font.name = "Calibri"
        return zone

    def nouvelle_diapositive(self, titre: str, sous_titre: str | None = None, notes: str = ""):
        slide = self.presentation.slides.add_slide(self.modele)
        self.numero += 1
        self._zone(slide, 0, 0, LARGEUR, Inches(1.05), VERT_FONCE)
        self._zone(slide, 0, Inches(1.05), LARGEUR, Inches(0.06), VERT_CLAIR)
        self._texte(slide, Inches(0.55), Inches(0.2), Inches(11.4), Inches(0.7),
                    titre, taille=26, couleur=BLANC, gras=True)
        if sous_titre:
            self._texte(slide, Inches(0.6), Inches(1.22), Inches(12), Inches(0.4),
                        sous_titre, taille=14, couleur=GRIS, italique=True)

        # numéro de page
        self._texte(slide, Inches(12.4), Inches(7.0), Inches(0.7), Inches(0.3),
                    str(self.numero), taille=11, couleur=GRIS, alignement=PP_ALIGN.RIGHT)
        # pied de page
        self._texte(slide, Inches(0.55), Inches(7.0), Inches(8), Inches(0.3),
                    "Détection des maladies de la feuille de tomate — CNN & Transfer Learning",
                    taille=9, couleur=GRIS)
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
        return slide

    def puces(self, slide, elements: list[str], gauche=Inches(0.7), haut=Inches(1.8),
              largeur=Inches(11.9), hauteur=Inches(4.9), taille=17):
        zone = slide.shapes.add_textbox(gauche, haut, largeur, hauteur)
        cadre = zone.text_frame
        cadre.word_wrap = True
        for index, element in enumerate(elements):
            paragraphe = cadre.paragraphs[0] if index == 0 else cadre.add_paragraph()
            paragraphe.level = 0
            paragraphe.space_after = Pt(9)
            run = paragraphe.add_run()
            run.text = "▪  " + element
            run.font.size = Pt(taille)
            run.font.color.rgb = GRIS
            run.font.name = "Calibri"
        return zone

    def encadre(self, slide, gauche, haut, largeur, hauteur, titre, valeur, couleur=VERT):
        self._zone(slide, gauche, haut, largeur, hauteur, GRIS_CLAIR)
        self._zone(slide, gauche, haut, Inches(0.08), hauteur, couleur)
        self._texte(slide, gauche + Inches(0.25), haut + Inches(0.12), largeur - Inches(0.4),
                    Inches(0.5), valeur, taille=22, couleur=couleur, gras=True)
        self._texte(slide, gauche + Inches(0.25), haut + Inches(0.62), largeur - Inches(0.4),
                    Inches(0.5), titre, taille=11, couleur=GRIS)

    def image(self, slide, chemin: Path, gauche, haut, largeur=None, hauteur=None):
        if not chemin.exists():
            self._texte(slide, gauche, haut, Inches(4), Inches(0.5),
                        f"[image manquante : {chemin.name}]", taille=12, couleur=ROUGE)
            return None
        if largeur is not None and hauteur is None:
            return slide.shapes.add_picture(str(chemin), gauche, haut, width=largeur)
        if hauteur is not None and largeur is None:
            return slide.shapes.add_picture(str(chemin), gauche, haut, height=hauteur)
        return slide.shapes.add_picture(str(chemin), gauche, haut, width=largeur, height=hauteur)

    def tableau(self, slide, donnees: list[list[str]], gauche, haut, largeur, hauteur,
                taille=13):
        lignes, colonnes = len(donnees), len(donnees[0])
        forme = slide.shapes.add_table(lignes, colonnes, gauche, haut, largeur, hauteur)
        table = forme.table
        for index_ligne, ligne in enumerate(donnees):
            for index_colonne, valeur in enumerate(ligne):
                cellule = table.cell(index_ligne, index_colonne)
                cellule.text = valeur
                paragraphe = cellule.text_frame.paragraphs[0]
                paragraphe.font.size = Pt(taille)
                paragraphe.font.name = "Calibri"
                paragraphe.font.bold = index_ligne == 0
                paragraphe.font.color.rgb = BLANC if index_ligne == 0 else GRIS
                cellule.fill.solid()
                cellule.fill.fore_color.rgb = VERT_FONCE if index_ligne == 0 else BLANC
        return table

    def enregistrer(self, chemin: Path) -> Path:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        self.presentation.save(chemin)
        return chemin


def construire() -> Path:
    configuration = charger_json(config.TRAINING_CONFIG_PATH, defaut={}) or {}
    metriques = charger_json(config.METRICS_DIR / "evaluation.json", defaut={}) or {}
    manifeste = charger_json(config.DATASET_MANIFEST_PATH, defaut={}) or {}
    global_ = metriques.get("global", {})
    par_classe = metriques.get("par_classe", {})
    captures = RACINE / "docs" / "rapport" / "captures"
    figures = RACINE / "results" / "figures"

    d = Diaporama()

    # 1. Titre ---------------------------------------------------------
    slide = d.presentation.slides.add_slide(d.modele)
    d._zone(slide, 0, 0, LARGEUR, HAUTEUR, VERT_FONCE)
    d._zone(slide, Inches(0.9), Inches(2.55), Inches(2.0), Inches(0.07), VERT_CLAIR)
    d._texte(slide, Inches(0.9), Inches(1.1), Inches(11.5), Inches(0.5),
             "Projet universitaire de Deep Learning — Génie informatique",
             taille=16, couleur=VERT_CLAIR, gras=True)
    d._texte(slide, Inches(0.9), Inches(1.75), Inches(11.5), Inches(1.6),
             "Détection et classification des maladies\n"
             "de la feuille de tomate",
             taille=40, couleur=BLANC, gras=True)
    d._texte(slide, Inches(0.9), Inches(2.85), Inches(11.5), Inches(0.9),
             "Réseaux de neurones convolutifs et Transfer Learning — MobileNetV2, TensorFlow/Keras, Flask",
             taille=17, couleur=RGBColor(0xD1, 0xFA, 0xE5))
    d._texte(slide, Inches(0.9), Inches(4.3), Inches(6), Inches(1.4),
             "Présenté par : [Prénom NOM]\nEncadré par : [Nom de l'encadrant]\n"
             f"Année universitaire 2025 – 2026",
             taille=13, couleur=RGBColor(0xE5, 0xE7, 0xEB))
    d._texte(slide, Inches(8.2), Inches(4.3), Inches(4.4), Inches(1.6),
             f"Exactitude (test) : {pourcent(global_.get('accuracy', 0))}\n"
             f"F1-score macro : {pourcent(global_.get('f1_macro', 0))}\n"
             f"{len(config.CLASS_NAMES)} classes · {global_.get('nb_images', '—')} images de test",
             taille=14, couleur=VERT_CLAIR, gras=True, alignement=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = (
        "Bonjour. Je présente un système de détection des maladies de la feuille de tomate "
        "fondé sur un CNN pré-entraîné, MobileNetV2, adapté par Transfer Learning, et intégré "
        "dans une application Web Flask. (≈ 40 s)"
    )

    # 2. Plan ----------------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Plan de la présentation",
        notes="Annoncer le déroulé en 20 secondes.",
    )
    d.puces(slide, [
        "1. Contexte, problématique et objectifs",
        "2. Dataset PlantVillage et préparation des données",
        "3. Méthodologie : CNN, MobileNetV2 et Transfer Learning",
        "4. Entraînement et évaluation du modèle",
        "5. Résultats expérimentaux et analyse des erreurs",
        "6. Application Web et démonstration",
        "7. Limites, perspectives et conclusion",
    ], taille=18)

    # 3. Contexte / problématique --------------------------------------
    slide = d.nouvelle_diapositive(
        "Contexte et problématique",
        "Pourquoi automatiser la reconnaissance visuelle des maladies foliaires ?",
        notes=(
            "Insister sur l'enjeu agronomique, puis sur la difficulté du diagnostic visuel : "
            "symptômes proches entre alternariose et mildiou, besoin d'expertise, diagnostic tardif. "
            "(≈ 1 min)"
        ),
    )
    d.puces(slide, [
        "Les maladies foliaires de la tomate (alternariose, mildiou, moisissure) réduisent fortement les rendements.",
        "Le diagnostic visuel est tardif, subjectif et dépend d'un expert disponible.",
        "Les CNN apprennent automatiquement des caractéristiques visuelles discriminantes.",
        "Problématique : classer automatiquement une photo de feuille parmi 4 catégories, "
        "localement, en moins d'une seconde, avec une interface compréhensible.",
    ], taille=17)

    # 4. Objectifs ------------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Objectifs du projet",
        notes="Quatre objectifs : données, modèle, évaluation, application. (≈ 40 s)",
    )
    objectifs = [
        ("Données", "Découpage stratifié\nsans fuite (70/15/15)"),
        ("Modèle", "MobileNetV2\nTransfer Learning en 2 phases"),
        ("Évaluation", "Accuracy, F1, matrice\nde confusion, erreurs"),
        ("Application", "Flask + SQLite\ninterface en français"),
    ]
    for index, (titre, texte) in enumerate(objectifs):
        gauche = Inches(0.55 + index * 3.1)
        d._zone(slide, gauche, Inches(1.9), Inches(2.9), Inches(2.0), GRIS_CLAIR)
        d._zone(slide, gauche, Inches(1.9), Inches(2.9), Inches(0.08), VERT_CLAIR)
        d._texte(slide, gauche + Inches(0.2), Inches(2.1), Inches(2.5), Inches(0.4),
                 titre, taille=18, couleur=VERT_FONCE, gras=True)
        d._texte(slide, gauche + Inches(0.2), Inches(2.6), Inches(2.5), Inches(1.2),
                 texte, taille=12, couleur=GRIS)
    d.puces(slide, [
        "Contrainte : exécution locale, sans GPU, sans service payant, entièrement reproductible.",
    ], haut=Inches(4.4), hauteur=Inches(1.0), taille=16)

    # 5. Dataset --------------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Dataset : PlantVillage (sous-ensemble tomate)",
        "5 452 images couleur 256 × 256 — 4 classes — sources publiques",
        notes=(
            "Citer Hughes & Salathé (2015) et Mohanty et al. (2016). Mentionner le déséquilibre "
            "des classes (952 à 1 909 images) et le fait que les images sont prises en laboratoire. "
            "(≈ 1 min 15)"
        ),
    )
    d.image(slide, figures / "dataset_examples.png", Inches(0.5), Inches(1.75), largeur=Inches(8.2))
    d.tableau(slide, [
        ["Classe", "Images"],
        ["Alternariose", "1 000"],
        ["Mildiou", "1 909"],
        ["Moisissure", "952"],
        ["Saine", "1 591"],
        ["Total", "5 452"],
    ], Inches(9.1), Inches(1.8), Inches(3.7), Inches(2.6), taille=12)
    d.puces(slide, [
        "Déséquilibre : de 952 à 1 909 images par classe → poids de classes.",
        "14 doublons exacts détectés (MD5) et regroupés dans un même ensemble.",
    ], haut=Inches(4.9), largeur=Inches(8.6), hauteur=Inches(1.6), taille=14)

    # 6. Préparation ----------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Préparation des données",
        "Découpage stratifié, reproductible et sans fuite",
        notes=(
            "Souligner les trois garanties : split avant tout calcul, doublons regroupés, test "
            "jamais utilisé avant l'évaluation finale. (≈ 50 s)"
        ),
    )
    d.image(slide, figures / "dataset_distribution.png", Inches(0.5), Inches(1.7), largeur=Inches(8.4))
    d.tableau(slide, [
        ["Ensemble", "Images"],
        ["Entraînement", "3 811"],
        ["Validation", "821"],
        ["Test", "820"],
    ], Inches(9.3), Inches(1.9), Inches(3.4), Inches(1.8), taille=13)
    d.puces(slide, [
        "Graine fixée (42) et découpage par classe.",
        "Le test n'est chargé qu'au moment de l'évaluation.",
    ], haut=Inches(4.2), gauche=Inches(9.3), largeur=Inches(3.6), hauteur=Inches(1.8), taille=13)

    # 7. Prétraitement --------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Prétraitement et Data Augmentation",
        "RGB → 224 × 224 → normalisation [-1, 1] ; augmentation uniquement à l'entraînement",
        notes=(
            "Insister sur l'identité du prétraitement entre entraînement et inférence, vérifiée "
            "par un test automatisé. (≈ 50 s)"
        ),
    )
    d.image(slide, figures / "augmentation_examples.png", Inches(0.4), Inches(1.7), largeur=Inches(12.5))
    d.puces(slide, [
        "Rotation ±5 %, zoom ±10 %, retournement horizontal, luminosité ±15 %, translation ±5 %.",
        "Normalisation identique à mobilenet_v2.preprocess_input — vérifiée par un test.",
    ], haut=Inches(4.9), hauteur=Inches(1.6), taille=15)

    # 8. MobileNetV2 / TL -----------------------------------------------
    slide = d.nouvelle_diapositive(
        "Méthodologie : MobileNetV2 + Transfer Learning",
        "Base ImageNet gelée puis fine-tuning partiel — 2,4 M de paramètres",
        notes=(
            "Expliquer les convolutions séparables en profondeur et le principe du Transfer "
            "Learning : réutiliser les filtres génériques, remplacer la classification, puis "
            "ajuster finement. (≈ 1 min 30)"
        ),
    )
    d.image(slide, figures / "model_architecture.png", Inches(0.6), Inches(1.6), largeur=Inches(11.6))
    d.puces(slide, [
        "Phase 1 : base gelée → seuls 164 484 paramètres apprennent (Adam 1e-3).",
        "Phase 2 : 19 dernières couches dégelées, BatchNormalization gelées (Adam 1e-5).",
        "Justification : modèle léger, 70 ms par image sur CPU.",
    ], haut=Inches(4.8), hauteur=Inches(1.9), taille=15)

    # 9. Entraînement ---------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Configuration de l'entraînement",
        notes=(
            "Présenter les hyperparamètres et les garde-fous : Early Stopping, checkpoint, "
            "réduction du learning rate. Préciser le budget réduit (15 + 8 époques) sur CPU. "
            "(≈ 1 min)"
        ),
    )
    d.tableau(slide, [
        ["Paramètre", "Valeur"],
        ["Entrée", "224 × 224 × 3"],
        ["Batch size", "32"],
        ["Optimiseur", "Adam (1e-3 puis 1e-5)"],
        ["Perte", "Categorical crossentropy"],
        ["Époques", "15 + 8 (max : 30 + 15)"],
        ["Callbacks", "EarlyStopping, Checkpoint, ReduceLROnPlateau"],
        ["Graine", "42"],
    ], Inches(0.6), Inches(1.8), Inches(7.2), Inches(3.4), taille=13)
    d.encadre(slide, Inches(8.3), Inches(1.9), Inches(4.4), Inches(1.25),
              "Durée totale (CPU 2 cœurs)", "≈ 38 min", VERT)
    d.encadre(slide, Inches(8.3), Inches(3.35), Inches(4.4), Inches(1.25),
              "Meilleure accuracy de validation", pourcent(configuration.get("meilleure_val_accuracy", 0)), VERT)
    d.encadre(slide, Inches(8.3), Inches(4.8), Inches(4.4), Inches(1.25),
              "Paramètres entraînés (phase 1)", "164 484 / 2 422 468", ROUGE)

    # 10. Courbes -------------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Courbes d'apprentissage",
        f"Phase 1 : {pourcent(configuration.get('phase1', {}).get('meilleure_val_accuracy', 0))} — "
        f"phase 2 (fine-tuning) : {pourcent(configuration.get('meilleure_val_accuracy', 0))}",
        notes=(
            "Commenter la convergence très rapide dès l'époque 1 (86 %) grâce à ImageNet, puis "
            "le gain du fine-tuning. Mentionner la variabilité de la validation. (≈ 1 min)"
        ),
    )
    d.image(slide, figures / "training_curves.png", Inches(0.75), Inches(1.75), largeur=Inches(11.6))
    d._texte(slide, Inches(0.6), Inches(6.25), Inches(12), Inches(0.6),
             "Aucun sur-apprentissage marqué : écart entraînement / validation inférieur à 1 point en fin d'entraînement.",
             taille=13, couleur=VERT_FONCE, gras=True, alignement=PP_ALIGN.CENTER)

    # 11. Résultats globaux ---------------------------------------------
    slide = d.nouvelle_diapositive(
        "Résultats sur l'ensemble de test",
        f"{global_.get('nb_images', '—')} images jamais vues à l'entraînement",
        notes=(
            "Énoncer les 4 métriques et le nombre d'erreurs. Rappeler que l'accuracy seule ne "
            "suffit pas : d'où le F1 macro et la matrice de confusion. (≈ 1 min)"
        ),
    )
    d.encadre(slide, Inches(0.6), Inches(1.9), Inches(3.0), Inches(1.5),
              "Exactitude", pourcent(global_.get("accuracy", 0)), VERT)
    d.encadre(slide, Inches(3.8), Inches(1.9), Inches(3.0), Inches(1.5),
              "Précision (macro)", pourcent(global_.get("precision_macro", 0)), VERT)
    d.encadre(slide, Inches(7.0), Inches(1.9), Inches(3.0), Inches(1.5),
              "Rappel (macro)", pourcent(global_.get("recall_macro", 0)), VERT)
    d.encadre(slide, Inches(10.2), Inches(1.9), Inches(2.6), Inches(1.5),
              "F1-score (macro)", pourcent(global_.get("f1_macro", 0)), ROUGE)
    d.puces(slide, [
        f"{global_.get('nb_images', 0) - global_.get('nb_erreurs', 0)} images correctement classées "
        f"sur {global_.get('nb_images', 0)}.",
        f"{global_.get('nb_erreurs', 0)} erreurs ({pourcent(global_.get('nb_erreurs', 0) / max(global_.get('nb_images', 1), 1))}) "
        "concentrées sur alternariose ↔ mildiou.",
        "Performances homogènes : F1 macro (95,34 %) proche du F1 pondéré (95,82 %).",
    ], haut=Inches(3.8), hauteur=Inches(2.4), taille=16)

    # 12. Matrice de confusion ------------------------------------------
    slide = d.nouvelle_diapositive(
        "Matrice de confusion",
        notes=(
            "Lire la matrice par ligne : 88,7 % des alternarioses, 95,8 % des mildious, 96,5 % "
            "des moisissures, 100 % des feuilles saines. Les confusions sont biologiquement "
            "cohérentes. (≈ 1 min 15)"
        ),
    )
    d.image(slide, figures / "confusion_matrix.png", Inches(0.6), Inches(1.45), largeur=Inches(11.8))
    d._texte(slide, Inches(0.6), Inches(6.3), Inches(12.2), Inches(0.6),
             "Diagonale dominante · 15 confusions alternariose ↔ mildiou sur 34 erreurs totales.",
             taille=14, couleur=VERT_FONCE, gras=True, alignement=PP_ALIGN.CENTER)

    # 13. Par classe ----------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Performances par classe",
        notes=(
            "Souligner le rappel de 100 % sur les feuilles saines (pas de fausse alerte) et la "
            "difficulté de l'alternariose (F1 91,1 %). (≈ 1 min)"
        ),
    )
    d.image(slide, figures / "per_class_performance.png", Inches(0.5), Inches(1.7), largeur=Inches(8.0))
    lignes = [["Classe", "F1"]]
    for classe in config.CLASS_NAMES:
        lignes.append([
            nom_affichage(classe).split(" de la")[0],
            pourcent(par_classe.get(classe, {}).get("f1_score", 0), 1),
        ])
    d.tableau(slide, lignes, Inches(8.8), Inches(1.9), Inches(4.0), Inches(2.4), taille=13)
    d.puces(slide, [
        "Feuille saine : rappel 100 % — aucune fausse alerte.",
        "Alternariose : classe la plus difficile.",
    ], haut=Inches(4.6), gauche=Inches(8.8), largeur=Inches(4.0), hauteur=Inches(1.8), taille=13)

    # 14. Erreurs -------------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Analyse des erreurs",
        "Confiance moyenne : 97,5 % (correct) contre 70,2 % (erreur)",
        notes=(
            "Les erreurs sont souvent des cas ambigus : lésions peu contrastées, symptômes "
            "mélangés. Le modèle est moins confiant quand il se trompe. (≈ 1 min)"
        ),
    )
    d.image(slide, RACINE / "results" / "predictions" / "error_grid.png",
            Inches(0.5), Inches(1.6), hauteur=Inches(5.3))
    d.image(slide, figures / "confidence_distribution.png", Inches(7.7), Inches(1.8), largeur=Inches(5.1))
    d._texte(slide, Inches(7.7), Inches(4.45), Inches(5.1), Inches(1.5),
             "La confiance reste informative :\nelle chute sur les erreurs, mais le modèle est globalement sur-confiant.",
             taille=11, couleur=GRIS)

    # 15. Application Web -----------------------------------------------
    slide = d.nouvelle_diapositive(
        "Application Web : architecture",
        "Flask · SQLite · Bootstrap 5 · Chart.js · interface en français",
        notes=(
            "Décrire les couches : présentation, applicative, métier. Insister sur le chargement "
            "unique du modèle et sur la validation stricte des fichiers. (≈ 1 min)"
        ),
    )
    d.image(slide, captures / "01_tableau_de_bord.png", Inches(0.4), Inches(1.7), hauteur=Inches(5.0))
    d.puces(slide, [
        "5 pages : tableau de bord, analyse, historique, fiches maladies, à propos.",
        "Modèle chargé une seule fois au démarrage (singleton).",
        "Validation : extension, MIME, décodage Pillow, taille, renommage UUID.",
        "Historique SQLite : consultations et suppressions.",
        "API JSON : /api/analyser, /api/historique, /sante.",
        "44 tests automatisés (pytest), dont un parcours complet réel.",
    ], gauche=Inches(3.4), haut=Inches(1.8), largeur=Inches(9.4), hauteur=Inches(4.6), taille=14)

    # 16. Démonstration -------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Démonstration : analyse d'une feuille",
        "Téléversement → prédiction (~70 ms) → probabilités → historique",
        notes=(
            "Montrer l'analyse en direct si possible. Sinon commenter la capture : classe prédite, "
            "confiance, distribution des 4 probabilités, description et symptômes. (≈ 1 min 30)"
        ),
    )
    d.image(slide, captures / "05_apercu_image.png", Inches(0.4), Inches(1.7), hauteur=Inches(5.0))
    d.image(slide, captures / "07_resultat_entete.png", Inches(3.5), Inches(1.7), hauteur=Inches(5.0))
    d.puces(slide, [
        "Aperçu avant analyse, spinner pendant l'inférence.",
        "Résultat : classe, confiance, graphique des probabilités.",
        "Fiche complète enregistrée et consultable dans l'historique.",
    ], gauche=Inches(6.8), haut=Inches(1.9), largeur=Inches(6.2), hauteur=Inches(4.8), taille=14)

    # 17. Limites -------------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Limites et perspectives",
        notes=(
            "Assumer clairement les limites : quatre classes, images de laboratoire, "
            "sur-confiance, absence d'explicabilité. Présenter les perspectives concrètes. "
            "(≈ 1 min 15)"
        ),
    )
    d.puces(slide, [
        "Données : 4 classes sur 38, images de laboratoire, déséquilibre, test de 820 images.",
        "Modèle : une seule architecture, pas de validation croisée, probabilités mal calibrées.",
        "Application : pas d'authentification, serveur de développement Flask.",
        "Perspectives : Grad-CAM, détection d'inconnu, calibration (température scaling), "
        "validation croisée 5 plis, données de terrain, quantification TFLite.",
    ], taille=17)

    # 18. Conclusion ----------------------------------------------------
    slide = d.nouvelle_diapositive(
        "Conclusion",
        notes=(
            "Récapituler : objectif atteint, résultats mesurés, apports méthodologiques, "
            "posture honnête. Remercier et ouvrir sur les questions. (≈ 1 min)"
        ),
    )
    d.puces(slide, [
        "Un système complet : dataset → modèle → évaluation → application Web.",
        f"Résultats réels : {pourcent(global_.get('accuracy', 0))} d'accuracy, "
        f"F1 macro {pourcent(global_.get('f1_macro', 0))} sur {global_.get('nb_images', '—')} images de test.",
        "Transfer Learning efficace : 86 % de validation dès la première époque.",
        "Erreurs concentrées sur deux maladies visuellement proches.",
        "Outil expérimental d'aide à la reconnaissance visuelle — pas un diagnostic agricole.",
        "Perspectives : explicabilité, calibration, données de terrain, déploiement mobile.",
    ], taille=17)

    # 19. Merci ---------------------------------------------------------
    slide = d.presentation.slides.add_slide(d.modele)
    d._zone(slide, 0, 0, LARGEUR, HAUTEUR, VERT_FONCE)
    d._texte(slide, Inches(0.9), Inches(2.5), Inches(11.5), Inches(1.0),
             "Merci de votre attention", taille=40, couleur=BLANC, gras=True)
    d._texte(slide, Inches(0.9), Inches(3.6), Inches(11.5), Inches(1.0),
             "Questions et démonstration en direct", taille=20, couleur=VERT_CLAIR)
    d._texte(slide, Inches(0.9), Inches(5.2), Inches(11.5), Inches(1.4),
             "Références clés : Mohanty, Hughes & Salathé (2016) · Sandler et al. (2018) · "
             "Howard et al. (2017) · Selvaraju et al. (2017)",
             taille=12, couleur=RGBColor(0xD1, 0xFA, 0xE5))
    slide.notes_slide.notes_text_frame.text = "Remercier le jury et proposer la démonstration en direct."

    return d.enregistrer(SORTIE)


if __name__ == "__main__":
    chemin = construire()
    print(f"Présentation générée : {chemin}")
