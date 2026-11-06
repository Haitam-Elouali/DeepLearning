"""Capture les captures d'écran de l'application pour le rapport.

Outil annexe (non requis pour faire tourner le projet) : il utilise Playwright
avec Chromium headless pour photographier l'interface à une largeur de bureau.

Prérequis :
    pip install playwright
    playwright install chromium
    # l'application doit être démarrée sur http://127.0.0.1:3000

Usage :
    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "docs" / "rapport" / "captures"
BASE = "http://127.0.0.1:3000"
IMAGE_TEST = next(
    iter(
        sorted(
            (RACINE / "data" / "processed" / "test" / "Tomato___Late_blight").glob("*")
        )
    ),
    None,
)


def capturer() -> int:
    SORTIE.mkdir(parents=True, exist_ok=True)
    if IMAGE_TEST is None:
        print("Image de test introuvable : captures limitées aux pages statiques.")
    with sync_playwright() as pilote:
        navigateur = pilote.chromium.launch()
        page = navigateur.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        erreurs: list[str] = []
        page.on("console", lambda message: message.type == "error" and erreurs.append(message.text))
        page.on("pageerror", lambda erreur: erreurs.append(str(erreur)))

        # 1. Tableau de bord — haut de page
        page.goto(f"{BASE}/", wait_until="networkidle")
        page.wait_for_timeout(700)
        page.screenshot(path=str(SORTIE / "01_tableau_de_bord.png"))

        # 2. Tableau de bord — statistiques, classes et graphique
        page.evaluate("window.scrollTo(0, 620)")
        page.wait_for_timeout(900)
        page.screenshot(path=str(SORTIE / "02_statistiques_classes.png"))

        # 3. Tableau de bord — dernières analyses
        page.evaluate("document.querySelector('.table').scrollIntoView({block: 'center'})")
        page.wait_for_timeout(400)
        page.screenshot(path=str(SORTIE / "03_dernieres_analyses.png"))

        # 4. Page d'analyse avant téléversement
        page.goto(f"{BASE}/analyse", wait_until="networkidle")
        page.wait_for_timeout(400)
        page.screenshot(path=str(SORTIE / "04_page_analyse.png"))

        # 5. Analyse réelle : téléversement + prédiction
        if IMAGE_TEST is not None:
            page.set_input_files("#champImage", str(IMAGE_TEST))
            page.wait_for_timeout(600)
            page.screenshot(path=str(SORTIE / "05_apercu_image.png"))
            page.click("#boutonAnalyser")
            page.wait_for_selector("#resultats:not(.d-none)", timeout=30_000)
            page.wait_for_timeout(1200)  # animation du graphique
            page.screenshot(path=str(SORTIE / "06_resultat_analyse.png"))
            page.evaluate("document.getElementById('resultats').scrollIntoView({block: 'start'})")
            page.wait_for_timeout(600)
            page.screenshot(path=str(SORTIE / "07_resultat_detail.png"))

        # 8. Historique
        page.goto(f"{BASE}/historique", wait_until="networkidle")
        page.wait_for_timeout(400)
        page.screenshot(path=str(SORTIE / "08_historique.png"))

        # 9. Détail d'une analyse
        liens = page.query_selector_all("a[href^='/analyse/']")
        if liens:
            liens[0].click()
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(800)
            page.screenshot(path=str(SORTIE / "09_detail_analyse.png"))

        # 10. Fiches maladies
        page.goto(f"{BASE}/maladies", wait_until="networkidle")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SORTIE / "10_maladies.png"))

        # 11. À propos
        page.goto(f"{BASE}/a-propos", wait_until="networkidle")
        page.wait_for_timeout(400)
        page.screenshot(path=str(SORTIE / "11_a_propos.png"))

        # 12. Affichage mobile (vérification du responsive)
        mobile = navigateur.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        mobile.goto(f"{BASE}/analyse", wait_until="networkidle")
        mobile.wait_for_timeout(500)
        mobile.screenshot(path=str(SORTIE / "12_mobile_analyse.png"), full_page=True)

        navigateur.close()

    fichiers = sorted(SORTIE.glob("*.png"))
    print(f"{len(fichiers)} captures enregistrées dans {SORTIE}")
    for fichier in fichiers:
        print(f"  - {fichier.name} ({fichier.stat().st_size // 1024} Ko)")
    if erreurs:
        print("\nErreurs JavaScript détectées :")
        for erreur in erreurs:
            print(f"  ! {erreur}")
        return 1
    print("\nAucune erreur JavaScript.")
    return 0


if __name__ == "__main__":
    raise SystemExit(capturer())
