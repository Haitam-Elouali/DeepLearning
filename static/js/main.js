/* ==========================================================================
   TomatoScan — script principal
   Gère : téléversement (clic + glisser-déposer), aperçu, appel de l'API,
   affichage des résultats et suppression d'une analyse.
   ========================================================================== */
(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    initialiserTeleversement();
    initialiserSuppression();
  });

  /* ------------------------------------------------------------------ */
  /* Utilitaires                                                         */
  /* ------------------------------------------------------------------ */
  function formaterPourcentage(valeur, decimales) {
    const nombre = Number(valeur) || 0;
    return nombre.toFixed(decimales === undefined ? 1 : decimales).replace(".", ",") + " %";
  }

  function afficherMessage(conteneur, texte, type) {
    if (!conteneur) return;
    conteneur.className = "alert alert-" + (type || "success") + " mt-3";
    conteneur.textContent = texte;
    conteneur.classList.remove("d-none");
  }

  /* ------------------------------------------------------------------ */
  /* Téléversement et analyse                                            */
  /* ------------------------------------------------------------------ */
  function initialiserTeleversement() {
    const zone = document.getElementById("zoneDepot");
    const champ = document.getElementById("champImage");
    if (!zone || !champ) return; // page sans formulaire d'analyse

    const boutonAnalyser = document.getElementById("boutonAnalyser");
    const boutonReset = document.getElementById("boutonReinitialiser");
    const contenuDepot = document.getElementById("contenuDepot");
    const contenuApercu = document.getElementById("contenuApercu");
    const apercu = document.getElementById("apercu");
    const infosFichier = document.getElementById("infosFichier");
    const erreur = document.getElementById("erreurAnalyse");
    const spinner = document.getElementById("spinnerAnalyse");
    const icone = document.getElementById("iconeAnalyse");
    const texteBouton = document.getElementById("texteAnalyse");
    const blocAttente = document.getElementById("blocAttente");
    const resultats = document.getElementById("resultats");

    const tailleMax = Number(zone.dataset.tailleMax || 8 * 1024 * 1024);
    const typesAcceptes = ["image/jpeg", "image/png"];
    let fichierCourant = null;
    let urlApercu = null;
    let graphique = null;

    function libererApercu() {
      if (urlApercu) {
        URL.revokeObjectURL(urlApercu);
        urlApercu = null;
      }
    }

    function reinitialiser() {
      fichierCourant = null;
      champ.value = "";
      libererApercu();
      apercu.removeAttribute("src");
      contenuApercu.classList.add("d-none");
      contenuDepot.classList.remove("d-none");
      infosFichier.classList.add("d-none");
      erreur.classList.add("d-none");
      resultats.classList.add("d-none");
      blocAttente.classList.remove("d-none");
      boutonAnalyser.disabled = true;
      boutonReset.disabled = true;
      if (graphique) {
        graphique.destroy();
        graphique = null;
      }
      zone.focus();
    }

    function validerFichier(fichier) {
      if (!fichier) return "Aucun fichier sélectionné.";
      const type = (fichier.type || "").toLowerCase();
      const nom = (fichier.name || "").toLowerCase();
      const extensionValide = /\.(jpe?g|png)$/.test(nom);
      if (!typesAcceptes.includes(type) && !extensionValide) {
        return "Format non supporté. Choisissez une image JPG, JPEG ou PNG.";
      }
      if (fichier.size > tailleMax) {
        return "Image trop volumineuse (maximum " + Math.round(tailleMax / (1024 * 1024)) + " Mo).";
      }
      return null;
    }

    function choisirFichier(fichier) {
      const probleme = validerFichier(fichier);
      erreur.classList.add("d-none");
      if (probleme) {
        afficherMessage(erreur, probleme, "danger");
        return;
      }

      fichierCourant = fichier;
      libererApercu();
      urlApercu = URL.createObjectURL(fichier);
      apercu.src = urlApercu;
      apercu.alt = "Aperçu de l'image : " + fichier.name;
      contenuDepot.classList.add("d-none");
      contenuApercu.classList.remove("d-none");

      infosFichier.innerHTML =
        "<i class='bi bi-file-earmark-image me-1'></i>" +
        escapeHtml(fichier.name) +
        " · " + Math.max(1, Math.round(fichier.size / 1024)) + " Ko";
      infosFichier.classList.remove("d-none");

      boutonAnalyser.disabled = false;
      boutonReset.disabled = false;
      resultats.classList.add("d-none");
      blocAttente.classList.remove("d-none");
    }

    function escapeHtml(texte) {
      const element = document.createElement("div");
      element.textContent = texte;
      return element.innerHTML;
    }

    function definirChargement(actif) {
      boutonAnalyser.disabled = actif;
      boutonReset.disabled = actif;
      zone.classList.toggle("chargement", actif);
      spinner.classList.toggle("d-none", !actif);
      icone.classList.toggle("d-none", actif);
      texteBouton.textContent = actif ? "Analyse en cours…" : "Lancer l'analyse";
    }

    async function analyser() {
      if (!fichierCourant) return;
      definirChargement(true);
      erreur.classList.add("d-none");

      const donnees = new FormData();
      donnees.append("image", fichierCourant, fichierCourant.name);

      try {
        const reponse = await fetch("/api/analyser", {
          method: "POST",
          body: donnees,
          headers: { "Accept": "application/json" }
        });

        let contenu = null;
        try {
          contenu = await reponse.json();
        } catch (invalide) {
          contenu = null;
        }

        if (!reponse.ok || !contenu || !contenu.succes) {
          const message = (contenu && contenu.erreur)
            ? contenu.erreur
            : "L'analyse a échoué (code " + reponse.status + "). Réessayez.";
          afficherMessage(erreur, message, "danger");
          return;
        }

        afficherResultat(contenu);
      } catch (probleme) {
        afficherMessage(
          erreur,
          "Impossible de contacter le serveur. Vérifiez que l'application est démarrée.",
          "danger"
        );
      } finally {
        definirChargement(false);
      }
    }

    function afficherResultat(reponse) {
      const analyse = reponse.analyse;

      document.getElementById("resultatClasse").textContent = analyse.classe_nom;
      document.getElementById("resultatResume").textContent = analyse.resume;
      document.getElementById("resultatConfiance").textContent = formaterPourcentage(analyse.confiance_pct, 1);
      document.getElementById("resultatDuree").textContent = analyse.duree_ms + " ms";

      const barre = document.getElementById("barreConfianceRemplissage");
      barre.style.width = Math.max(2, analyse.confiance_pct) + "%";
      barre.style.background = analyse.saine
        ? "linear-gradient(90deg,#15803d,#22c55e)"
        : (analyse.confiance_pct < 60
            ? "linear-gradient(90deg,#b45309,#f59e0b)"
            : "linear-gradient(90deg,#b91c1c,#ef4444)");

      const badgeCategorie = document.getElementById("resultatCategorie");
      badgeCategorie.textContent = analyse.categorie;
      badgeCategorie.style.background = analyse.couleur;
      badgeCategorie.style.color = "#fff";

      document.getElementById("resultatNiveau").textContent = "Gravité indicative : " + analyse.niveau;
      document.getElementById("resultatDescription").textContent = analyse.description;

      const listeSymptomes = document.getElementById("resultatSymptomes");
      listeSymptomes.innerHTML = "";
      (analyse.symptomes || []).forEach(function (symptome) {
        const item = document.createElement("li");
        item.textContent = symptome;
        listeSymptomes.appendChild(item);
      });

      const lien = document.getElementById("lienDetail");
      lien.href = analyse.detail_url;

      // Liste ordonnée des probabilités
      const liste = document.getElementById("listeProbabilites");
      liste.innerHTML = "";
      reponse.probabilites.forEach(function (proba, index) {
        const item = document.createElement("li");
        const entete = document.createElement("div");
        entete.className = "d-flex justify-content-between small";
        entete.innerHTML =
          "<span>" + escapeHtml(proba.nom_fr) + "</span>" +
          "<span class='fw-semibold'>" + formaterPourcentage(proba.pourcentage, 1) + "</span>";
        const barreProba = document.createElement("div");
        barreProba.className = "progress";
        barreProba.style.height = "6px";
        const remplissage = document.createElement("div");
        remplissage.className = "progress-bar";
        remplissage.style.width = Math.max(1, proba.pourcentage) + "%";
        remplissage.style.background = index === 0 ? proba.couleur : "#cbd5c0";
        barreProba.appendChild(remplissage);
        item.appendChild(entete);
        item.appendChild(barreProba);
        liste.appendChild(item);
      });

      dessinerGraphique(reponse.probabilites);

      blocAttente.classList.add("d-none");
      resultats.classList.remove("d-none");
      resultats.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function dessinerGraphique(probabilites) {
      const canvas = document.getElementById("graphiqueProbabilites");
      if (!canvas || typeof Chart === "undefined") return;
      if (graphique) graphique.destroy();

      graphique = new Chart(canvas, {
        type: "bar",
        data: {
          labels: probabilites.map(function (p) { return p.nom_fr; }),
          datasets: [{
            label: "Probabilité (%)",
            data: probabilites.map(function (p) { return p.pourcentage; }),
            backgroundColor: probabilites.map(function (p) { return p.couleur; }),
            borderRadius: 6,
            maxBarThickness: 28
          }]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 600 },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: function (contexte) {
                  return formaterPourcentage(contexte.parsed.x, 1);
                }
              }
            }
          },
          scales: {
            x: {
              beginAtZero: true,
              max: 100,
              grid: { color: "rgba(0,0,0,0.06)" },
              ticks: { callback: function (valeur) { return valeur + " %"; } }
            },
            y: { grid: { display: false } }
          }
        }
      });
    }

    /* --- Événements --------------------------------------------------- */
    zone.addEventListener("click", function () { champ.click(); });
    zone.addEventListener("keydown", function (evenement) {
      if (evenement.key === "Enter" || evenement.key === " ") {
        evenement.preventDefault();
        champ.click();
      }
    });

    champ.addEventListener("change", function () {
      if (champ.files && champ.files.length) choisirFichier(champ.files[0]);
    });

    ["dragenter", "dragover"].forEach(function (type) {
      zone.addEventListener(type, function (evenement) {
        evenement.preventDefault();
        zone.classList.add("survol");
      });
    });

    ["dragleave", "drop"].forEach(function (type) {
      zone.addEventListener(type, function (evenement) {
        evenement.preventDefault();
        zone.classList.remove("survol");
      });
    });

    zone.addEventListener("drop", function (evenement) {
      const fichiers = evenement.dataTransfer && evenement.dataTransfer.files;
      if (fichiers && fichiers.length) choisirFichier(fichiers[0]);
    });

    boutonAnalyser.addEventListener("click", analyser);
    boutonReset.addEventListener("click", reinitialiser);
  }

  /* ------------------------------------------------------------------ */
  /* Suppression d'une analyse                                           */
  /* ------------------------------------------------------------------ */
  function initialiserSuppression() {
    const boutons = document.querySelectorAll(".bouton-supprimer");
    if (!boutons.length) return;

    const modalElement = document.getElementById("modalSuppression");
    const boutonConfirmer = document.getElementById("confirmerSuppression");
    const cible = document.getElementById("nomSuppression");
    const zoneMessage = document.getElementById("messageAction");
    let identifiant = null;
    let redirection = null;
    let modal = null;

    if (modalElement && typeof bootstrap !== "undefined") {
      modal = new bootstrap.Modal(modalElement);
    }

    boutons.forEach(function (bouton) {
      bouton.addEventListener("click", function () {
        identifiant = bouton.dataset.id;
        redirection = bouton.dataset.redirection || null;
        if (cible) cible.textContent = bouton.dataset.nom || ("#" + identifiant);

        if (modal) {
          modal.show();
        } else if (window.confirm("Supprimer définitivement cette analyse ?")) {
          supprimer();
        }
      });
    });

    if (boutonConfirmer) {
      boutonConfirmer.addEventListener("click", function () {
        supprimer();
      });
    }

    async function supprimer() {
      if (!identifiant) return;
      if (boutonConfirmer) boutonConfirmer.disabled = true;

      try {
        const reponse = await fetch("/api/historique/" + identifiant, {
          method: "DELETE",
          headers: { "Accept": "application/json" }
        });
        const contenu = await reponse.json().catch(function () { return null; });

        if (!reponse.ok || !contenu || !contenu.succes) {
          afficherMessage(zoneMessage, "La suppression a échoué.", "danger");
          zoneMessage.classList.remove("d-none");
          return;
        }

        if (modal) modal.hide();

        if (redirection) {
          window.location.href = redirection;
          return;
        }

        const ligne = document.getElementById("ligne-" + identifiant);
        if (ligne) {
          ligne.style.transition = "opacity 0.25s ease";
          ligne.style.opacity = "0";
          setTimeout(function () { ligne.remove(); }, 250);
        }
        afficherMessage(zoneMessage, "Analyse supprimée avec succès.", "success");
        zoneMessage.classList.remove("d-none");
      } catch (probleme) {
        afficherMessage(zoneMessage, "Impossible de contacter le serveur.", "danger");
        zoneMessage.classList.remove("d-none");
      } finally {
        if (boutonConfirmer) boutonConfirmer.disabled = false;
        identifiant = null;
      }
    }
  }
})();
