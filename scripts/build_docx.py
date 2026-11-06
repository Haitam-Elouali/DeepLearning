"""Convertit un document Markdown en fichier Word (.docx) mis en forme.

Ce convertisseur couvre volontairement un sous-ensemble du Markdown, suffisant
pour les livrables du projet : titres, paragraphes, listes, tableaux, images
avec légende, citations, gras/italique/code et sauts de page explicites.

Usage :
    python scripts/build_docx.py docs/rapport/rapport_final.md docs/rapport/rapport_final.docx
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

VERT = RGBColor(0x16, 0x65, 0x34)
GRIS = RGBColor(0x4B, 0x55, 0x63)

MOTIF_IMAGE = re.compile(r"^!\[(?P<legende>[^\]]*)\]\((?P<chemin>[^)]+)\)\s*$")
MOTIF_TABLEAU = re.compile(r"^\|.*\|\s*$")
MOTIF_SEPARATEUR_TABLEAU = re.compile(r"^\|[\s:|-]+\|\s*$")


# ---------------------------------------------------------------------------
# Mise en forme du document
# ---------------------------------------------------------------------------
def configurer_styles(document: Document) -> None:
    """Définit la police, les marges et les styles de titres."""
    for section in document.sections:
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.2)

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for nom, taille, couleur in (
        ("Title", 26, VERT),
        ("Heading 1", 18, VERT),
        ("Heading 2", 14, GRIS),
        ("Heading 3", 12, GRIS),
    ):
        style = document.styles[nom]
        style.font.name = "Calibri"
        style.font.size = Pt(taille)
        style.font.color.rgb = couleur
        style.font.bold = True
        style.paragraph_format.space_before = Pt(14)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True


def ajouter_pied_de_page(document: Document) -> None:
    """Ajoute un pied de page avec titre et numéro de page."""
    for section in document.sections:
        pied = section.footer.paragraphs[0]
        pied.alignment = WD_ALIGN_PARAGRAPH.CENTER
        texte = pied.add_run("Détection des maladies de la feuille de tomate — CNN & Transfer Learning  ·  page ")
        texte.font.size = Pt(8)
        texte.font.color.rgb = GRIS

        champ = OxmlElement("w:fldSimple")
        champ.set(qn("w:instr"), "PAGE")
        run = OxmlElement("w:r")
        propriete = OxmlElement("w:rPr")
        taille = OxmlElement("w:sz")
        taille.set(qn("w:val"), "16")
        propriete.append(taille)
        run.append(propriete)
        texte_champ = OxmlElement("w:t")
        texte_champ.text = "1"
        run.append(texte_champ)
        champ.append(run)
        pied._p.append(champ)


def mettre_en_forme_texte(paragraphe, texte: str) -> None:
    """Interprète **gras**, *italique* et `code` dans un paragraphe."""
    morceaux = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", texte)
    for morceau in morceaux:
        if not morceau:
            continue
        if morceau.startswith("**") and morceau.endswith("**"):
            run = paragraphe.add_run(morceau[2:-2])
            run.bold = True
        elif morceau.startswith("*") and morceau.endswith("*") and len(morceau) > 2:
            run = paragraphe.add_run(morceau[1:-1])
            run.italic = True
        elif morceau.startswith("`") and morceau.endswith("`"):
            run = paragraphe.add_run(morceau[1:-1])
            run.font.name = "Consolas"
            run.font.color.rgb = VERT
            run.font.size = Pt(9.5)
        else:
            paragraphe.add_run(morceau)


# ---------------------------------------------------------------------------
# Éléments de contenu
# ---------------------------------------------------------------------------
def ajouter_tableau(document: Document, lignes: list[str], base: Path) -> None:
    """Construit un tableau Word à partir des lignes Markdown."""
    cellules = [
        [cellule.strip() for cellule in ligne.strip().strip("|").split("|")] for ligne in lignes
    ]
    cellules = [ligne for ligne in cellules if not all(re.fullmatch(r"[-: ]*", c) for c in ligne)]
    if not cellules:
        return

    nombre_colonnes = max(len(ligne) for ligne in cellules)
    tableau = document.add_table(rows=len(cellules), cols=nombre_colonnes)
    tableau.style = "Light Grid Accent 1"
    tableau.alignment = WD_TABLE_ALIGNMENT.CENTER

    for index_ligne, ligne in enumerate(cellules):
        for index_colonne in range(nombre_colonnes):
            texte = ligne[index_colonne] if index_colonne < len(ligne) else ""
            cellule = tableau.cell(index_ligne, index_colonne)
            cellule.text = ""
            paragraphe = cellule.paragraphs[0]
            mettre_en_forme_texte(paragraphe, texte)
            for run in paragraphe.runs:
                run.font.size = Pt(9.5)
                if index_ligne == 0:
                    run.bold = True
    document.add_paragraph()


def ajouter_image(document: Document, chemin: Path, legende: str) -> None:
    """Insère une image centrée avec sa légende."""
    if not chemin.exists():
        paragraphe = document.add_paragraph()
        run = paragraphe.add_run(f"[Figure manquante : {chemin.name}]")
        run.italic = True
        run.font.color.rgb = GRIS
        return

    paragraphe = document.add_paragraph()
    paragraphe.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraphe.add_run()
    largeur = Cm(15.5)
    try:
        from PIL import Image

        with Image.open(chemin) as image:
            ratio = image.height / image.width
        if ratio > 1.05:  # image portrait : limiter la hauteur
            largeur = Cm(11)
    except Exception:  # noqa: BLE001 - la largeur par défaut suffit
        pass
    run.add_picture(str(chemin), width=largeur)

    if legende:
        paragraphe_legende = document.add_paragraph()
        paragraphe_legende.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_legende = paragraphe_legende.add_run(legende)
        run_legende.italic = True
        run_legende.font.size = Pt(9)
        run_legende.font.color.rgb = GRIS


def convertir(source: Path, destination: Path) -> Path:
    """Convertit le fichier Markdown en document Word."""
    document = Document()
    configurer_styles(document)
    ajouter_pied_de_page(document)

    lignes = source.read_text(encoding="utf-8").splitlines()
    base = source.parent
    index = 0
    premiere_section = True

    while index < len(lignes):
        ligne = lignes[index].rstrip()

        if not ligne.strip():
            index += 1
            continue

        if ligne.strip() == "[[PAGEBREAK]]":
            document.add_page_break()
            index += 1
            continue

        # Tableau
        if MOTIF_TABLEAU.match(ligne):
            bloc = []
            while index < len(lignes) and MOTIF_TABLEAU.match(lignes[index].strip()):
                bloc.append(lignes[index].strip())
                index += 1
            ajouter_tableau(document, bloc, base)
            continue

        # Image
        correspondance = MOTIF_IMAGE.match(ligne.strip())
        if correspondance:
            chemin_image = (base / correspondance.group("chemin")).resolve()
            ajouter_image(document, chemin_image, correspondance.group("legende"))
            index += 1
            continue

        # Titres
        if ligne.startswith("# "):
            if premiere_section:
                titre = document.add_paragraph(style="Title")
                mettre_en_forme_texte(titre, ligne[2:])
                titre.alignment = WD_ALIGN_PARAGRAPH.CENTER
                premiere_section = False
            else:
                document.add_heading(ligne[2:], level=0)
            index += 1
            continue

        if ligne.startswith("## "):
            document.add_heading(ligne[3:], level=1)
            index += 1
            continue

        if ligne.startswith("### "):
            document.add_heading(ligne[4:], level=2)
            index += 1
            continue

        if ligne.startswith("#### "):
            document.add_heading(ligne[5:], level=3)
            index += 1
            continue

        # Citation (les lignes consécutives sont regroupées en un paragraphe)
        if ligne.startswith("> "):
            bloc_citation = [ligne[2:].strip()]
            index += 1
            while index < len(lignes) and lignes[index].startswith(">"):
                bloc_citation.append(lignes[index].lstrip("> ").strip())
                index += 1

            paragraphe = document.add_paragraph()
            paragraphe.paragraph_format.left_indent = Cm(0.8)
            paragraphe.paragraph_format.right_indent = Cm(0.5)
            mettre_en_forme_texte(paragraphe, " ".join(bloc_citation))
            for run in paragraphe.runs:
                run.italic = True
                run.font.color.rgb = GRIS
                run.font.size = Pt(10)
            continue

        # Listes
        if re.match(r"^[-*] ", ligne):
            paragraphe = document.add_paragraph(style="List Bullet")
            mettre_en_forme_texte(paragraphe, ligne[2:])
            index += 1
            continue

        if re.match(r"^\d+\. ", ligne):
            paragraphe = document.add_paragraph(style="List Number")
            mettre_en_forme_texte(paragraphe, re.sub(r"^\d+\. ", "", ligne))
            index += 1
            continue

        # Paragraphe (regroupement des lignes consécutives)
        bloc_paragraphe = [ligne.strip()]
        index += 1
        while (
            index < len(lignes)
            and lignes[index].strip()
            and not lignes[index].startswith(("#", "|", ">", "- ", "* ", "!["))
            and not re.match(r"^\d+\. ", lignes[index])
            and lignes[index].strip() != "[[PAGEBREAK]]"
        ):
            bloc_paragraphe.append(lignes[index].strip())
            index += 1

        paragraphe = document.add_paragraph()
        paragraphe.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        mettre_en_forme_texte(paragraphe, " ".join(bloc_paragraphe))

    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)
    return destination


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    if not source.exists():
        raise SystemExit(f"Fichier source introuvable : {source}")
    resultat = convertir(source, destination)
    mots = len(source.read_text(encoding="utf-8").split())
    print(f"Document Word généré : {resultat} ({mots} mots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
