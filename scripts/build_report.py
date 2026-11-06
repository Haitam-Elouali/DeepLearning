"""Assemble le rapport final et le convertit en document Word.

Le rapport est rédigé en trois parties (`_partie1.md`, `_partie2.md`,
`_partie3.md`) afin de rester facile à relire et à corriger, puis assemblé en un
document unique avant conversion. La conversion Markdown → DOCX est assurée par
`scripts/build_docx.py`.

Usage :
    python scripts/build_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DOSSIER = RACINE / "docs" / "rapport"
PARTIES = ["_partie1.md", "_partie2.md", "_partie3.md"]
ASSEMBLE = DOSSIER / "rapport_final.md"
DOCX = DOSSIER / "rapport_final.docx"

sys.path.insert(0, str(RACINE / "scripts"))
from build_docx import convertir  # noqa: E402


def assembler() -> Path:
    """Concatène les parties du rapport en un seul document Markdown."""
    morceaux: list[str] = []
    for nom in PARTIES:
        chemin = DOSSIER / nom
        if not chemin.exists():
            raise SystemExit(f"Partie manquante : {chemin}")
        morceaux.append(chemin.read_text(encoding="utf-8").strip())

    contenu = "\n\n".join(morceaux) + "\n"
    ASSEMBLE.write_text(contenu, encoding="utf-8")
    return ASSEMBLE


def main() -> int:
    source = assembler()
    convertir(source, DOCX)

    texte = source.read_text(encoding="utf-8")
    print(f"Rapport Markdown assemblé : {source} ({len(texte.split())} mots)")
    print(f"Document Word généré     : {DOCX} ({DOCX.stat().st_size / 1024:.0f} Ko)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
