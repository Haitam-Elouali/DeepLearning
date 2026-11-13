"""Convertit le rapport Markdown en LaTeX et compile le PDF.

Le rapport est rédigé en Markdown (``docs/rapport/_partie*.md`` assemblés en
``rapport_final.md`` par ``scripts/build_report.py``). Ce script en produit une
version LaTeX autonome, avec page de titre, sommaire, tableaux et figures, puis
lance ``xelatex`` deux fois (la deuxième passe sert à stabiliser le sommaire et
les numéros de page).

Les informations nominatives restent des **espaces réservés** : elles sont
déclarées en tête du fichier ``.tex`` dans quatre macros qu'il suffit de
compléter.

Usage :
    python scripts/build_latex.py              # conversion + compilation
    python scripts/build_latex.py --no-pdf     # conversion seule
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SOURCE = RACINE / "docs" / "rapport" / "rapport_final.md"
DOSSIER_SORTIE = RACINE / "docs" / "rapport" / "latex"
FICHIER_TEX = DOSSIER_SORTIE / "rapport_final.tex"
FICHIER_PDF = DOSSIER_SORTIE / "rapport_final.pdf"

#: Remplacements caractère par caractère (une seule passe : les accolades
#: introduites par une substitution ne sont jamais ré-échappées).
REMPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "—": "---",
    "–": "--",
    "«": r"\guillemotleft{}",
    "»": r"\guillemotright{}",
    "→": r"$\rightarrow$",
    "≈": r"$\approx$",
    "±": r"$\pm$",
    "×": r"$\times$",
    "°": r"\textdegree{}",
    "²": r"\textsuperscript{2}",
    "₂": r"\textsubscript{2}",
    "ᵉ": r"\textsuperscript{e}",
    "µ": r"\textmu{}",
    "§": r"\S{}",
}

#: Fractions de largeur de ligne utilisées pour les colonnes de tableau.
LARGEURS = {
    2: [0.28, 0.58],
    3: [0.22, 0.31, 0.33],
    4: [0.24, 0.13, 0.27, 0.22],
    5: [0.22, 0.16, 0.16, 0.16, 0.16],
    6: [0.18, 0.136, 0.136, 0.136, 0.136, 0.136],
}

CAPTION = re.compile(r"^\*(Tableau|Figure)\s+[\dA-Z]+\s*[—–-]")
LISTE_PUCE = re.compile(r"^(\s*)[-*]\s+(.*)$")
LISTE_NUM = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")


# ---------------------------------------------------------------------------
# Échappement et mise en forme en ligne
# ---------------------------------------------------------------------------
def echapper(texte: str) -> str:
    """Échappe les caractères spéciaux LaTeX et les symboles Unicode isolés."""
    return "".join(REMPLACEMENTS.get(caractere, caractere) for caractere in texte)


def en_ligne(texte: str) -> str:
    """Convertit le Markdown en ligne (gras, italique, code, liens)."""
    texte = texte.replace("\\\\", "\\")

    codes: list[str] = []

    def _code(correspondance: re.Match) -> str:
        codes.append(correspondance.group(1))
        return f"\x00{len(codes) - 1}\x00"

    texte = re.sub(r"`([^`]+)`", _code, texte)

    liens: list[tuple[str, str]] = []

    def _lien(correspondance: re.Match) -> str:
        liens.append((correspondance.group(1), correspondance.group(2)))
        return f"\x01{len(liens) - 1}\x01"

    texte = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", _lien, texte)
    texte = echapper(texte)
    texte = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", texte)
    texte = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\\emph{\1}", texte)

    for index, code in enumerate(codes):
        texte = texte.replace(
            f"\x00{index}\x00",
            r"\texttt{" + echapper(code).replace("/", r"/\allowbreak{}").replace("\\_", r"\_\allowbreak{}") + "}",
        )
    for index, (libelle, url) in enumerate(liens):
        texte = texte.replace(
            f"\x01{index}\x01", r"\href{" + url + "}{" + echapper(libelle) + "}"
        )
    return texte


def nettoyer_caption(ligne: str) -> str:
    """Transforme ``*Tableau 17 — ...*`` en texte de légende."""
    contenu = ligne.strip().strip("*").strip()
    return en_ligne(contenu)


# ---------------------------------------------------------------------------
# Blocs
# ---------------------------------------------------------------------------
def convertir_tableau(lignes: list[str], legende: str | None) -> list[str]:
    """Convertit un tableau Markdown en ``longtable`` LaTeX."""
    lignes = [ligne.strip() for ligne in lignes if ligne.strip()]
    entete = [cellule.strip() for cellule in lignes[0].strip("|").split("|")]
    corps = [
        [cellule.strip() for cellule in ligne.strip("|").split("|")]
        for ligne in lignes[2:]
    ]
    colonnes = max(len(entete), max((len(ligne) for ligne in corps), default=0))
    fractions = LARGEURS.get(colonnes, [0.9 / colonnes] * colonnes)

    specification = "@{}" + "".join(
        r">{\raggedright\arraybackslash}p{" + f"{fraction:.2f}" + r"\linewidth}"
        for fraction in fractions[:colonnes]
    ) + "@{}"

    sortie: list[str] = []
    if legende:
        sortie.append(r"\begin{center}\small\itshape " + legende + r"\end{center}")
    sortie.append(r"{\footnotesize")
    sortie.append(r"\begin{longtable}{" + specification + "}")
    sortie.append(r"\toprule")
    sortie.append(
        " & ".join(
            r"\textbf{" + en_ligne(cellule) + "}" for cellule in entete
        )
        + r" \\"
    )
    sortie.append(r"\midrule")
    sortie.append(r"\endhead")
    sortie.append(r"\bottomrule")
    sortie.append(r"\endlastfoot")
    for ligne in corps:
        cellules = [en_ligne(cellule) for cellule in ligne]
        cellules += [""] * (colonnes - len(cellules))
        sortie.append(" & ".join(cellules[:colonnes]) + r" \\")
    sortie.append(r"\end{longtable}")
    sortie.append(r"}")
    sortie.append("")
    return sortie


def convertir_figure(chemin: str, legende: str | None, dossier: Path) -> list[str]:
    """Convertit ``![alt](chemin)`` en environnement ``figure``."""
    cible = (dossier / chemin).resolve()
    try:
        relatif = os.path.relpath(cible, DOSSIER_SORTIE).replace("\\", "/")
    except ValueError:  # lecteurs différents sous Windows
        relatif = cible.as_posix()

    largeur, hauteur = 0.68, 0.42

    sortie = [
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=" + f"{largeur:.2f}" + r"\linewidth,height="
        + f"{hauteur:.2f}" + r"\textheight,keepaspectratio]{" + relatif + "}",
    ]
    if legende:
        sortie.append(r"\par\smallskip{\small\itshape " + legende + "}")
    sortie.append(r"\end{figure}")
    sortie.append("")
    return sortie


def convertir_liste(elements: list[tuple[int, str, str]]) -> list[str]:
    """Convertit une liste (niveau, type, texte) en environnements imbriqués."""
    sortie: list[str] = []
    pile: list[tuple[int, str]] = []

    for niveau, genre, texte in elements:
        while pile and pile[-1][0] > niveau:
            sortie.append(r"\end{" + pile.pop()[1] + "}")
        if not pile or pile[-1][0] < niveau:
            environnement = "itemize" if genre == "puce" else "enumerate"
            sortie.append(r"\begin{" + environnement + "}")
            pile.append((niveau, environnement))
        sortie.append(r"\item " + en_ligne(texte))

    while pile:
        sortie.append(r"\end{" + pile.pop()[1] + "}")
    sortie.append("")
    return sortie


# ---------------------------------------------------------------------------
# En-tête du document (page de titre)
# ---------------------------------------------------------------------------
def analyser_entete(lignes: list[str]) -> dict[str, str]:
    """Extrait le titre, le sous-titre et les champs nominatifs du préambule."""
    infos = {"titre": "", "sous_titre": "", "label": "", "auteur": "[Prénom NOM]",
             "encadrant": "[Nom de l'encadrant]",
             "etablissement": "[Nom de l'université / de l'école]",
             "annee": "2025 -- 2026"}

    for ligne in lignes:
        texte = ligne.strip()
        if not texte:
            continue
        if texte.startswith("# ") and not infos["titre"]:
            infos["titre"] = texte[2:].strip()
            continue
        if "Réalisé par" in texte:
            infos["auteur"] = texte.split(":", 1)[-1].strip().strip("*")
            continue
        if "Encadré par" in texte:
            infos["encadrant"] = texte.split(":", 1)[-1].strip().strip("*")
            continue
        if "Établissement" in texte:
            infos["etablissement"] = texte.split(":", 1)[-1].strip().strip("*")
            continue
        if "Année universitaire" in texte:
            infos["annee"] = texte.split(":", 1)[-1].strip().strip("*")
            continue
        if texte.startswith("**") and texte.endswith("**") and not infos["sous_titre"]:
            infos["sous_titre"] = texte.strip("*").strip()
            continue
        if not infos["label"] and texte.startswith("Projet universitaire"):
            infos["label"] = texte
    return infos


def valeurs_existantes(chemin: Path) -> dict[str, str]:
    """Relit les valeurs déjà saisies dans les macros du ``.tex`` existant.

    Cela évite de perdre le nom de l'auteur ou de l'encadrant lorsqu'on
    régénère le document depuis le Markdown.
    """
    if not chemin.exists():
        return {}
    texte = chemin.read_text(encoding="utf-8", errors="replace")
    valeurs: dict[str, str] = {}
    for macro in ("author", "supervisor", "institution", "year"):
        correspondance = re.search(
            r"\\newcommand\{\\report" + macro + r"\}\{(.*)\}", texte
        )
        if correspondance:
            valeurs[macro] = correspondance.group(1).strip()
    return valeurs


def preambule(infos: dict[str, str]) -> list[str]:
    """Construit le préambule LaTeX, la page de titre et le sommaire."""
    titre = en_ligne(infos["titre"])
    sous_titre = en_ligne(infos["sous_titre"])
    return [
        "% =====================================================================",
        "% Rapport généré par scripts/build_latex.py — ne pas éditer à la main.",
        "% Les quatre macros ci-dessous contiennent les espaces réservés à",
        "% compléter avant le rendu final.",
        "% =====================================================================",
        r"\newcommand{\reportauthor}{" + infos["auteur"] + "}",
        r"\newcommand{\reportsupervisor}{" + infos["encadrant"] + "}",
        r"\newcommand{\reportinstitution}{" + infos["etablissement"] + "}",
        r"\newcommand{\reportyear}{" + infos["annee"] + "}",
        "",
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage{fontspec}",
        r"\usepackage[french]{babel}",
        r"\usepackage[a4paper,top=1.8cm,bottom=1.8cm,left=2.3cm,right=2.0cm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{longtable,booktabs,array}",
        r"\usepackage{float}",
        r"\usepackage{enumitem}",
        r"\usepackage{xcolor}",
        r"\usepackage{titlesec}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{amsmath}",
        r"\usepackage[hidelinks]{hyperref}",
        "",
        r"\definecolor{tomate}{HTML}{15803D}",
        r"\definecolor{ardoise}{HTML}{475569}",
        "",
        r"\titleformat{\section}{\Large\bfseries\color{tomate}}{\thesection}{0.8em}{}",
        r"\titleformat{\subsection}{\large\bfseries\color{ardoise}}{\thesubsection}{0.6em}{}",
        r"\titlespacing*{\section}{0pt}{14pt}{6pt}",
        r"\titlespacing*{\subsection}{0pt}{10pt}{4pt}",
        r"\setlist{itemsep=1pt,topsep=3pt,parsep=0pt,leftmargin=1.3em}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{2pt}",
        r"\setlength{\emergencystretch}{3em}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        "",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        r"\fancyhead[L]{\small\itshape Détection des maladies de la feuille de tomate}",
        r"\fancyhead[R]{\small\thepage}",
        r"\renewcommand{\headrulewidth}{0.4pt}",
        "",
        r"\begin{document}",
        "",
        r"\begin{titlepage}",
        r"\centering",
        r"{\small " + en_ligne(infos["label"]) + r"\par}",
        r"\vspace{2.4cm}",
        r"{\LARGE\bfseries " + titre + r"\par}",
        r"\vspace{1.2cm}",
        r"{\large\itshape " + sous_titre + r"\par}",
        r"\vfill",
        r"\begin{tabular}{@{}r@{\hspace{1em}}l@{}}",
        r"Réalisé par : & \reportauthor \\[4pt]",
        r"Encadré par : & \reportsupervisor \\[4pt]",
        r"Établissement : & \reportinstitution \\[4pt]",
        r"Année universitaire : & \reportyear \\",
        r"\end{tabular}",
        r"\vfill",
        r"{\small\itshape Projet universitaire de Deep Learning — document généré depuis le dépôt du projet.\par}",
        r"\end{titlepage}",
        "",
        r"\pagenumbering{roman}",
        r"\setcounter{tocdepth}{1}",
        r"\tableofcontents",
        r"\clearpage",
        r"\pagenumbering{arabic}",
        "",
    ]


# ---------------------------------------------------------------------------
# Conversion du corps
# ---------------------------------------------------------------------------
def convertir_corps(lignes: list[str], dossier: Path) -> list[str]:
    """Parcourt le Markdown et produit le corps LaTeX."""
    sortie: list[str] = []
    index = 0
    total = len(lignes)
    ignorer_sommaire = False
    premier_titre = True
    compteur_section = 0

    while index < total:
        brute = lignes[index]
        ligne = brute.strip()

        # --- sauts de page et titres -----------------------------------
        if "[[PAGEBREAK]]" in ligne:
            sortie.append(r"\clearpage")
            sortie.append("")
            index += 1
            continue

        if ligne.startswith("# ") and premier_titre:
            premier_titre = False  # déjà utilisée sur la page de titre
            index += 1
            while index < total and "[[PAGEBREAK]]" not in lignes[index]:
                index += 1
            index += 1
            continue

        if ligne.startswith("## "):
            titre = ligne[3:].strip()
            if titre.lower().startswith("sommaire"):
                ignorer_sommaire = True
                index += 1
                continue
            ignorer_sommaire = False
            titre = re.sub(r"^\d+\.\s*", "", titre)
            if titre.lower().startswith("annexe"):
                sortie.append(r"\section*{" + en_ligne(titre) + "}")
                sortie.append(r"\addcontentsline{toc}{section}{" + en_ligne(titre) + "}")
            else:
                compteur_section += 1
                sortie.append(r"\section{" + en_ligne(titre) + "}")
                sortie.append(
                    r"\label{sec:" + str(compteur_section) + "}"
                    + "% " + titre.replace("\n", " ")
                )
            sortie.append("")
            index += 1
            continue

        if ligne.startswith("### "):
            titre = re.sub(r"^\d+\.\d+\.?\s*", "", ligne[4:].strip())
            sortie.append(r"\subsection{" + en_ligne(titre) + "}")
            sortie.append("")
            index += 1
            continue

        if ignorer_sommaire:
            index += 1
            continue

        # --- tableau ---------------------------------------------------
        if ligne.startswith("|"):
            bloc: list[str] = []
            while index < total and lignes[index].strip().startswith("|"):
                bloc.append(lignes[index])
                index += 1
            legende = None
            suivant = index
            while suivant < total and not lignes[suivant].strip():
                suivant += 1
            if suivant < total and CAPTION.match(lignes[suivant].strip()):
                legende = nettoyer_caption(lignes[suivant])
                index = suivant + 1
            sortie.extend(convertir_tableau(bloc, legende))
            continue

        # --- figure ----------------------------------------------------
        correspondance = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", ligne)
        if correspondance:
            alt, chemin = correspondance.group(1), correspondance.group(2)
            index += 1
            legende = alt or None
            suivant = index
            while suivant < total and not lignes[suivant].strip():
                suivant += 1
            if suivant < total and CAPTION.match(lignes[suivant].strip()):
                legende = nettoyer_caption(lignes[suivant])
                index = suivant + 1
            sortie.extend(convertir_figure(chemin, legende, dossier))
            continue

        # --- citation --------------------------------------------------
        if ligne.startswith(">"):
            bloc = []
            while index < total and lignes[index].strip().startswith(">"):
                bloc.append(lignes[index].strip().lstrip(">").strip())
                index += 1
            sortie.append(r"\begin{quote}\small")
            sortie.append(en_ligne(" ".join(bloc)))
            sortie.append(r"\end{quote}")
            sortie.append("")
            continue

        # --- listes ----------------------------------------------------
        if LISTE_PUCE.match(brute) or LISTE_NUM.match(brute):
            elements: list[tuple[int, str, str]] = []
            while index < total:
                candidate = lignes[index]
                puce = LISTE_PUCE.match(candidate)
                numero = LISTE_NUM.match(candidate)
                if puce:
                    niveau = len(puce.group(1)) // 2
                    elements.append((niveau, "puce", puce.group(2).strip()))
                    index += 1
                elif numero:
                    niveau = len(numero.group(1)) // 2
                    elements.append((niveau, "numero", numero.group(2).strip()))
                    index += 1
                elif candidate.strip() and elements and candidate[:1].isspace():
                    # continuation d'un item sur la ligne suivante
                    niveau, genre, texte = elements[-1]
                    elements[-1] = (niveau, genre, texte + " " + candidate.strip())
                    index += 1
                else:
                    break
            sortie.extend(convertir_liste(elements))
            continue

        # --- paragraphe ------------------------------------------------
        if not ligne:
            index += 1
            continue
        paragraphe: list[str] = []
        while index < total:
            candidate = lignes[index].strip()
            if (
                not candidate
                or candidate.startswith(("#", "|", ">", "!", "```"))
                or "[[PAGEBREAK]]" in candidate
                or LISTE_PUCE.match(lignes[index])
                or LISTE_NUM.match(lignes[index])
            ):
                break
            paragraphe.append(candidate)
            index += 1
        if paragraphe:
            sortie.append(en_ligne(" ".join(paragraphe)))
            sortie.append("")

    return sortie


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
def trouver_xelatex() -> str | None:
    """Cherche xelatex dans le PATH puis dans l'installation MiKTeX standard."""
    trouve = shutil.which("xelatex")
    if trouve:
        return trouve
    candidats = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/MiKTeX/miktex/bin/x64/xelatex.exe",
        Path("C:/Program Files/MiKTeX/miktex/bin/x64/xelatex.exe"),
    ]
    for candidat in candidats:
        if candidat.exists():
            return str(candidat)
    return None


def compiler(tex: Path) -> tuple[bool, str]:
    """Compile deux fois le document pour stabiliser le sommaire."""
    moteur = trouver_xelatex()
    if moteur is None:
        return False, "xelatex introuvable : installez MiKTeX ou TeX Live."

    sortie = ""
    for passe in (1, 2):
        resultat = subprocess.run(
            [
                moteur,
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex.name,
            ],
            cwd=tex.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        sortie += f"\n--- passe {passe} (code {resultat.returncode}) ---\n"
        sortie += (resultat.stdout or "")[-4000:]
        if resultat.returncode != 0:
            return False, sortie + (resultat.stderr or "")
    return True, sortie


def main() -> int:
    parseur = argparse.ArgumentParser(description="Rapport Markdown → LaTeX → PDF.")
    parseur.add_argument("--no-pdf", action="store_true", help="Ne pas compiler le PDF.")
    arguments = parseur.parse_args()

    if not SOURCE.exists():
        raise SystemExit(
            f"Rapport introuvable : {SOURCE}\nLancez d'abord : python scripts/build_report.py"
        )

    texte = SOURCE.read_text(encoding="utf-8")
    lignes = texte.splitlines()
    dossier = SOURCE.parent

    if "[[PAGEBREAK]]" in texte:
        position = lignes.index("[[PAGEBREAK]]")
        entete, corps = lignes[:position], lignes[position + 1 :]
    else:
        entete, corps = lignes[:12], lignes[12:]

    infos = analyser_entete(entete)

    # Conserve les valeurs déjà saisies par l'utilisateur dans le .tex.
    correspondance = {
        "author": "auteur",
        "supervisor": "encadrant",
        "institution": "etablissement",
        "year": "annee",
    }
    for macro, cle in correspondance.items():
        valeur = valeurs_existantes(FICHIER_TEX).get(macro)
        if valeur and valeur != infos[cle]:
            infos[cle] = valeur

    document = preambule(infos) + convertir_corps(corps, dossier) + [r"\end{document}", ""]

    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    FICHIER_TEX.write_text("\n".join(document), encoding="utf-8")
    print(f"LaTeX écrit : {FICHIER_TEX} ({len(document)} lignes)")

    if arguments.no_pdf:
        return 0

    succes, journal = compiler(FICHIER_TEX)
    if not succes:
        (DOSSIER_SORTIE / "compilation.log").write_text(journal, encoding="utf-8")
        print("Échec de la compilation. Extrait :")
        print(journal[-2500:])
        return 1

    if FICHIER_PDF.exists():
        pages = re.search(r"Output written on .*?\((\d+) pages?", journal)
        taille = FICHIER_PDF.stat().st_size / 1024
        print(
            f"PDF généré   : {FICHIER_PDF} "
            f"({taille:.0f} Ko, {pages.group(1) if pages else '?'} pages)"
        )
        afficher_pagination(FICHIER_TEX.parent)
    return 0


def afficher_pagination(dossier: Path) -> None:
    """Affiche la page de départ de chaque section (utile pour la mise en page)."""
    aux = dossier / "rapport_final.aux"
    if not aux.exists():
        return
    entrees = re.findall(
        r"\\newlabel\{sec:(\d+)\}\{\{([^}]*)\}\{([^}]*)\}",
        aux.read_text(encoding="utf-8", errors="replace"),
    )
    if not entrees:
        return
    print("Pagination des sections :")
    for numero, _valeur, page in entrees:
        print(f"  section {numero:>2} -> page {page}")


if __name__ == "__main__":
    sys.exit(main())
