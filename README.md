# Détection et classification des maladies des feuilles — PlantVillage (38 classes)

**Application expérimentale de reconnaissance visuelle par CNN et Transfer Learning.**

Projet universitaire de Deep Learning — TensorFlow/Keras, MobileNetV2, Flask.
Interface entièrement en français.

> ⚠️ **Avertissement** : cet outil est un projet pédagogique d'aide à la reconnaissance
> visuelle. Il ne constitue pas un diagnostic agricole et ne remplace ni l'observation
> d'un agronome ni une analyse de laboratoire.

---

## 1. Ce que fait le projet

| Étape | Détail |
| --- | --- |
| Téléversement | une image JPG ou PNG, 8 Mo maximum |
| Prétraitement | RGB, redimensionnement 224 × 224 (Pillow), normalisation [-1, 1] |
| Prédiction | MobileNetV2 + Transfer Learning, **38 classes**, sortie softmax |
| Résultat | classe prédite, confiance, durée, 5 probabilités les plus élevées |
| Confidentialité | aucune image ni aucun résultat conservé (traitement en mémoire) |

**Classes reconnues** : les **38 classes** du dataset PlantVillage (couleur), soit
14 cultures (tomate, pomme de terre, maïs, pommier, vigne, pêcher, poivron, fraise,
cerise, agrume, framboise, soja, courge, myrtille) et leurs maladies foliaires.
Les quatre classes de tomate ont un libellé français (Alternariose précoce, Mildiou,
Moisissure des feuilles, Feuille saine) ; les autres sont affichées sous la forme
« Culture — Maladie ».

---

## 2. Structure du dépôt

```text
DeepLearning/
├── app.py                      # application Flask : 2 routes (page + analyse)
├── config.py                   # chemins, hyperparamètres, libellés des classes
├── requirements.txt            # dépendances (bornes compatibles)
├── requirements-lock.txt       # versions exactes validées de l'expérience
├── README.md
│
├── data/
│   ├── raw/PlantVillage/       # images brutes (non versionnées, 38 dossiers)
│   ├── processed/              # train/validation/test + dataset_manifest.json
│   └── class_names.json        # ordre des classes du modèle
│
├── models/
│   ├── best_model.keras        # modèle final (livrable)
│   └── training_config.json    # hyperparamètres et résultats de l'entraînement
│
├── notebooks/
│   ├── 01_exploration.ipynb    # dataset, distribution, qualité, prétraitement
│   └── training.ipynb          # architecture, hyperparamètres, courbes, résultats
├── training/
│   ├── download_dataset.py     # récupère les 38 classes couleur (~955 Mo)
│   ├── prepare_data.py         # contrôle qualité + split stratifié sans fuite
│   ├── explore_data.py         # figures d'exploration (dataset, augmentation, modèle)
│   ├── train.py                # MobileNetV2 : phase 1 + fine-tuning
│   ├── evaluate.py             # métriques, figures, analyse d'erreurs
│   └── predict.py              # inférence en ligne de commande
│
├── utils/
│   ├── preprocessing.py        # pipeline d'image partagé train/inférence
│   ├── predictor.py            # chargement unique du modèle + prédiction
│   └── helpers.py              # validation du fichier, libellés, formatage
│
├── templates/                  # base.html, index.html
├── static/css/style.css        # feuille de style unique
│
├── results/
│   ├── metrics.json            # livrable : accuracy, precision, recall, F1 (38 classes)
│   ├── classification_report.csv  # livrable : rapport détaillé par classe
│   ├── training_curves.png     # livrable : loss entraînement / validation
│   ├── confusion_matrix.png    # livrable : matrice de confusion (38 classes)
│   ├── metrics/                # rapport texte, résumé, historique, qualité des données
│   ├── figures/                # performances par classe, distribution de la confiance
│   └── predictions/            # prédictions du test, erreurs
│
├── tests/                      # pytest : prétraitement, doublons, cohérence,
│                               #   résultats, utilitaires, application
│
├── scripts/                    # génération des livrables (rapport, fiche, slides)
└── docs/
    ├── rapport/                # rapport Markdown (3 parties = 8 chapitres), Word, LaTeX/PDF
    │   └── assets/             # logos de la page de titre (facultatifs)
    ├── fiche/                  # fiche expérimentale
    └── presentation/           # diaporama PowerPoint
```

Chaque module a une responsabilité unique : `config.py` centralise chemins et
hyperparamètres, `utils/preprocessing.py` garantit un prétraitement identique entre
entraînement et inférence, `utils/predictor.py` charge le modèle une seule fois, et
`app.py` ne contient que les routes et le rendu.

---

## 3. Installation

Prérequis : Python 3.10 à 3.12 (environnement validé : **3.12.10**) et `git` (pour
le téléchargement du dataset).

```bash
git clone https://github.com/Haitam-Elouali/DeepLearning.git
cd DeepLearning

python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

pip install -r requirements.txt    # ou requirements-lock.txt (versions exactes)
```

Vérification immédiate de l'installation :

```bash
python -c "import tensorflow as tf, flask, sklearn; print(tf.__version__, flask.__version__)"
python -m pytest -q                # doit se terminer sans échec
```

> Sur une machine équipée d'un GPU NVIDIA, remplacer `tensorflow-cpu` par
> `tensorflow` dans `requirements.txt` (même API, mêmes fichiers `.keras`).

---

## 4. Dataset

- **Nom** : PlantVillage Dataset (dossier `raw/color`, images couleur 256 × 256)
- **Source** : <https://github.com/spMohanty/PlantVillage-Dataset>
- **Publication** : Hughes & Salathé (2015), arXiv:1511.08060 ; Mohanty, Hughes &
  Salathé (2016), *Frontiers in Plant Science*, 7, 1419.
- **Images** : 54 305 photographies couleur, **38 classes**, 14 cultures, ≈ 955 Mo.

| Ensemble | Images | Part réelle |
| --- | --- | --- |
| Entraînement (70 %) | 37 999 | 69,97 % |
| Validation (15 %) | 8 153 | 15,01 % |
| Test (15 %) | 8 153 | 15,01 % |
| **Total** | **54 305** | 100 % |

Le dépôt d'origine ne contient pas de fichier de licence explicite : les images sont
utilisées dans un cadre strictement académique, ne sont pas redistribuées et ne sont pas
versionnées ici.

```bash
python training/download_dataset.py   # clone partiel des 38 dossiers couleur (~955 Mo)
python training/prepare_data.py       # contrôle qualité + split 70/15/15
python training/explore_data.py       # figures d'exploration (optionnel)
```

`prepare_data.py` vérifie chaque image, réalise un découpage **stratifié** avec la
graine 42, puis écrit `data/processed/dataset_manifest.json` (effectifs par classe et
par ensemble).

Règles du découpage (`construire_decoupage`) :

- les objectifs sont calculés sur le **nombre réel d'images** de chaque classe, et non
  sur le nombre de groupes de doublons : 70 % train, 15 % validation, 15 % test ;
- les images identiques d'une même classe forment un **bloc insécable**, affecté d'un
  seul tenant à l'ensemble qui présente le plus grand déficit relatif — un doublon exact
  ne peut donc jamais apparaître dans deux ensembles ;
- si une classe est trop petite pour respecter 15 %, le **minimum de 30 images par
  ensemble** (`config.MIN_IMAGES_PER_SPLIT`) s'applique en priorité ; les proportions de
  cette classe s'écartent alors légèrement de 70/15/15 et sont **documentées dans le
  manifeste** (`classes_rares_ecartees_des_ratios`) ;
- la graine 42 rend le découpage reproductible : relancer `--force` redonne exactement
  les mêmes fichiers.

La détection des doublons est **globale** : les empreintes MD5 sont calculées sur
l'ensemble du dataset, toutes classes confondues, *avant* le découpage (le dictionnaire
d'empreintes n'est jamais réinitialisé au changement de classe). Politique appliquée :

| Cas | Traitement |
| --- | --- |
| Images identiques dans une même classe | regroupées et placées dans le **même** ensemble (jamais séparées) |
| Images identiques étiquetées dans **deux classes différentes** | **conflit d'annotation** : écartées du découpage et signalées, aucune classe n'est choisie arbitrairement |
| Images différentes | conservées telles quelles |
| Quasi-doublons (empreinte perceptuelle dHash, distance de Hamming ≤ 2) | **signalés** pour inspection, jamais supprimés |

Le rapport complet (groupes, fichiers, conflits, quasi-doublons) est écrit dans
`results/metrics/qualite_donnees.json`. Pour le recalculer seul, sans toucher au
découpage existant :

```bash
python training/prepare_data.py --analyse-doublons
```

Résultat mesuré sur cette machine :

```text
Classe                                       Train  Valid.   Test   Brut
Apple___Apple_scab                             442      94      94    630
...
Potato___healthy                                92      30      30    152
Tomato___Tomato_Yellow_Leaf_Curl_Virus        3749     804     804   5357
TOTAL                                        37999    8153    8153  54305
```

Constats du contrôle qualité : **0 image illisible**, **21 groupes de doublons exacts**
(21 images dupliquées, toutes dans une même classe), **0 conflit d'annotation** entre
classes, **196 groupes de quasi-doublons** signalés (dont 56 inter-classes) et un
déséquilibre de **152 à 5 507 images** par classe (rapport 36).

Proportions réellement obtenues : **69,97 % / 15,01 % / 15,01 %**. Une seule classe est
concernée par la règle des 30 images : `Potato___healthy` (152 images → 92 / 30 / 30,
soit 61 / 20 / 20 %) ; le manifeste l'indique explicitement.

Vérification de reproductibilité : relancer `prepare_data.py --force` redonne exactement
37 999 / 8 153 / 8 153 images, classe par classe (test automatisé :
`tests/test_dataset.py`).

---

## 5. Entraînement

```bash
python training/train.py
```

Paramètres (valeurs réellement utilisées, centralisées dans `config.py`) :

| Paramètre | Valeur |
| --- | --- |
| Taille d'entrée | 224 × 224 × 3 |
| Batch size | 32 |
| Phase 1 (base gelée) | 30 époques, Early Stopping (patience 5), Checkpoint |
| Phase 2 (fine-tuning) | 15 époques, 30 dernières couches examinées, Adam 1e-5 |
| Poids de classes | activés (compensent le déséquilibre, poids de 0,26 à 10,87) |
| Graine | 42 |
| Durée observée | 498 min (phase 1) + 267 min (phase 2) ≈ **12 h 45** sur 4 threads CPU |

Déroulement :

- **phase 1** : base gelée, Adam 1e-3, Early Stopping + Checkpoint sur la meilleure
  validation ; meilleure accuracy de validation **94,75 %** (époque 27) ;
- **phase 2** : on recharge ce meilleur modèle, on examine les 30 dernières couches de
  MobileNetV2 et on dégèle toutes celles qui ne sont pas des BatchNormalization (elles
  restent gelées, ce qui stabilise l'apprentissage avec de petits lots) ; on recompile à
  1e-5 et on continue les époques. Bilan réel : **19 couches dégelées**, dont **10
  porteuses de poids** effectivement réentraînées (les autres sont des activations),
  **134 couches à poids restées gelées**, **52 BatchNormalization sur 52 gelées** ;
  meilleure accuracy de validation **97,01 %** (époque 45).

Ces valeurs ne sont pas recopiées à la main : elles sont calculées par
`resume_fine_tuning()` à partir du modèle réel et écrites dans
`models/training_config.json` (bloc `phase2_fine_tuning.fine_tuning`).

**Pourquoi deux phases** : en phase 1 la base MobileNetV2 (pré-entraînée sur ImageNet)
reste gelée et seule la petite tête de classification apprend (168 870 paramètres
entraînables sur 2 426 854) ; en phase 2 les dernières couches sont dégelées avec un
learning rate très faible pour les adapter aux textures de feuilles, sans détruire les
filtres déjà utiles.

**Data Augmentation** : rotation, zoom, retournement horizontal, luminosité et
translation sont appliqués uniquement aux images d'entraînement, dans le pipeline de
chargement. Ces couches ne font pas partie du modèle sauvegardé : l'inférence reste donc
« RGB → 224 × 224 → normalisation [-1, 1] », identique à l'entraînement.

---

## 6. Évaluation

```bash
python training/evaluate.py
```

Livrables produits à la racine de `results/` :

| Fichier | Contenu |
| --- | --- |
| `results/metrics.json` | accuracy, precision, recall et F1 (macro et pondérés), métriques par classe, matrice de confusion |
| `results/classification_report.csv` | precision / recall / F1 / support pour chacune des 38 classes |
| `results/training_curves.png` | loss et accuracy par époque (entraînement vs validation) |
| `results/confusion_matrix.png` | matrice de confusion (effectifs et proportions) sur le test |

Fichiers complémentaires : `results/metrics/classification_report.txt`,
`results/metrics/evaluation_summary.md`, `results/figures/per_class_performance.png`,
`results/figures/confidence_distribution.png`, `results/predictions/*`.

Toutes les métriques sont calculées à partir des prédictions réelles du modèle sur
l'ensemble de test (8 153 images, jamais utilisées pendant l'entraînement). Aucune
valeur n'est simulée ni recopiée d'une publication. Les deux matrices de confusion
(`confusion_matrix.png` en effectifs, `confusion_matrix_normalized.png` en pourcentage
par classe réelle) sont produites à 300 dpi à partir de ces mêmes prédictions.

---

## 7. Application Web

```bash
python app.py
# ou
flask --app app run
```

Ouvrir <http://127.0.0.1:5000>, choisir une image, cliquer sur « Analyser l'image ».

L'application tient en **deux routes** :

| Route | Méthode | Rôle |
| --- | --- | --- |
| `/` | GET | formulaire de téléversement |
| `/analyser` | POST | validation, prédiction, affichage du résultat |

Le résultat affiche l'image analysée, la classe prédite, la confiance, la durée de
l'inférence et les **cinq probabilités les plus élevées** (sur 38 classes, tout afficher
rendrait la page illisible). Aucune image n'est enregistrée : le fichier est traité en
mémoire. Il n'y a ni base de données, ni historique, ni API JSON, ni JavaScript.

Si `models/best_model.keras` est absent, l'application démarre quand même et affiche un
message clair au lieu de planter.

---

## 8. Prédiction en ligne de commande

```bash
python training/predict.py chemin/vers/feuille.jpg
python training/predict.py --dossier mes_photos/ --json resultats.json
```

---

## 9. Tests

```bash
pytest -q
```

77 tests couvrent :

- **app** : page d'accueil, analyse réelle de bout en bout (image → prédiction →
  affichage), refus des fichiers invalides, erreurs 400/404/413 ;
- **prétraitement** : conversion RGB, redimensionnement, normalisation, équivalence
  exacte avec `mobilenet_v2.preprocess_input` ;
- **pipeline** : un seul redimensionnement dans tout le projet (Pillow, le même à
  l'entraînement, en validation, en test et à l'inférence — TensorFlow et Pillow ne
  donnant pas exactement le même résultat bilinéaire), comparaison image par image des
  tenseurs `tf.data` avec le prétraitement d'inférence, absence de double normalisation,
  encodage one-hot, métadonnées du fine-tuning ;
- **découpage** : total conservé, effectifs par ensemble, proportions 70/15/15,
  absence de doublon exact entre ensembles, reproductibilité, règle des 30 images ;
- **doublons** : images identiques dans une classe, images identiques dans deux classes
  (conflit d'annotation), images différentes, homonymes de contenu différent, détection
  des quasi-doublons, non-séparation des doublons lors du découpage ;
- **cohérence** : 38 classes uniques, 14 cultures, ordre des classes identique dans
  `config.py`, `class_names.json`, le manifeste et les métadonnées ; absence de toute
  référence à un modèle à quatre classes ;
- **résultats** : métriques recalculées depuis `predictions_test.csv`, supports par
  classe conformes au manifeste, matrice de confusion cohérente, 45 époques dans
  l'historique, chiffres du README et de la fiche identiques au JSON ;
- **utilitaires** : extensions autorisées, refus des faux types MIME, des contenus
  invalides, des images trop petites ou trop grandes, formatage des libellés.

---

## 10. Résultats expérimentaux

Résultats mesurés sur l'ensemble de test indépendant (**8 153 images, 38 classes**),
avec le modèle `models/best_model.keras` :

| Métrique (test) | Valeur |
| --- | --- |
| Exactitude (accuracy) | **97,90 %** |
| Précision macro | 97,26 % |
| Rappel macro | 97,77 % |
| F1-score macro | **97,44 %** |
| F1-score pondéré | 97,91 % |
| Images mal classées | 171 (2,10 %) |
| Meilleure exactitude de validation | 97,01 % (époque 45) |
| Temps d'inférence | ≈ 110 ms par image (CPU) |

Ces valeurs sont **recalculées** par `python training/evaluate.py` à partir des
prédictions réelles sur le jeu de test (`results/metrics.json`,
`results/classification_report.csv`) ; le test `tests/test_resultats.py` vérifie
qu'elles correspondent au README et à la fiche expérimentale.

Comparaison avec la littérature : Mohanty et al. (2016) rapportent plus de 99 % sur les
38 classes avec AlexNet ou GoogLeNet et un entraînement bien plus long. Notre modèle,
2,4 millions de paramètres seulement, atteint 97,90 % : c'est un compromis
précision / coût assumé.

Classes les mieux reconnues (F1 = 100 %) : pommier (rouille de cèdre), cerise (oïdium),
maïs (rouille commune), maïs sain, raisin sain, orange (HLB), framboise saine, courge
(oïdium), fraise (brûlure des feuilles) et fraise saine. **Dix-neuf classes** dépassent
99 % de F1-score.

Classes les plus difficiles :

| Classe | Précision | Rappel | F1-score | Support |
| --- | --- | --- | --- | --- |
| Maïs — tache grise (Cercospora) | 77,32 % | 97,40 % | 86,21 % | 77 |
| Alternariose précoce (tomate) | 95,35 % | 82,00 % | 88,17 % | 150 |
| Tomate — tache cible | 88,64 % | 92,42 % | 90,49 % | 211 |
| Maïs — brûlure nordique | 98,44 % | 85,14 % | 91,30 % | 148 |
| Tomate — tétranyque | 89,26 % | 96,02 % | 92,51 % | 251 |

Les 171 erreurs se concentrent sur quelques paires de classes visuellement proches
(brûlure nordique → tache grise : 22 cas ; virus YLCV → brûlure bactérienne : 10 cas ;
tache cible → tétranyque : 10 cas ; alternariose → tache cible : 9 cas ; alternariose →
mildiou : 6 cas). La confiance moyenne est de 98,69 % pour les prédictions correctes
contre 71,31 % pour les erreurs.

---

## 11. Reproductibilité et limites

**Reproductibilité** : graine fixée (42), découpage stratifié déterministe, détection des
doublons, hyperparamètres centralisés et journalisés, figures générées par script.

**Limites principales** :

- images de laboratoire (fond uniforme) ; le modèle n'a jamais vu de conditions réelles ;
- déséquilibre extrême entre classes (152 à 5 507 images) : les classes rares reposent sur
  30 à 54 images de test seulement ;
- similarité visuelle entre certaines maladies (maïs, tomate) : principale source de
  confusion ;
- aucune indication sur la gravité ou le stade de la maladie ;
- pas de détection d'image hors domaine : toute image est classée dans l'une des 38 classes ;
- un seul run d'entraînement, sans validation croisée.

---

## 12. Livrables universitaires

| Livrable | Emplacement |
| --- | --- |
| Code source (scripts + notebook) | ce dépôt |
| Modèle entraîné | `models/best_model.keras` + `models/training_config.json` |
| Métriques (tableau) | `results/metrics.json` et `results/classification_report.csv` |
| Courbes Training / Validation Loss | `results/training_curves.png` |
| Matrice de confusion | `results/confusion_matrix.png` |
| Rapport scientifique (Word) | `docs/rapport/rapport_final.docx` |
| Rapport scientifique (LaTeX + PDF) | `docs/rapport/latex/rapport_final.tex` et `rapport_final.pdf` |
| Fiche expérimentale | `docs/fiche/fiche_experimentale.docx` |
| Présentation orale | `docs/presentation/presentation_soutenance.pptx` |

Régénération des livrables :

```bash
python scripts/build_report.py          # rapport Markdown + Word
python scripts/build_latex.py           # rapport LaTeX + PDF (XeLaTeX)
python scripts/build_fiche.py           # fiche expérimentale
python scripts/build_presentation.py    # diaporama PowerPoint
```

Le rapport suit la présentation d'un rapport de projet de fin d'année (EMSI) : page de
titre avec le bloc EMSI en grand, sommaire, huit chapitres numérotés, bibliographie et
annexe. Le texte est **noir uniquement**, sans en-tête ni cadre coloré ; seuls les
numéros de page figurent en pied de page.

> **Logos** : déposez `logo_emsi.png` (ou `.jpg`, `.pdf`) dans
> `docs/rapport/assets/` — tous les fichiers `logo*` y sont insérés en grand sur la page
> de titre. Sans fichier, seul le bloc typographique « EMSI — École Marocaine des
> Sciences de l'Ingénieur » est affiché.
>
> L'auteur et l'année restent des espaces réservés dans les macros `\report...` en tête
> de `docs/rapport/latex/rapport_final.tex`. Les valeurs saisies sont conservées si le
> script est relancé ; sinon, compiler directement avec `xelatex rapport_final.tex`.
> La compilation demande XeLaTeX (MiKTeX ou TeX Live).

---

## 13. Sécurité et qualité

- extensions et types MIME contrôlés, contenu réellement décodé par Pillow ;
- taille maximale (8 Mo) et dimensions contrôlées (32 à 4 000 px) ;
- traitement en mémoire : aucune image écrite sur le disque, aucun chemin construit à
  partir du nom fourni par le client ;
- aucun chemin interne renvoyé au navigateur ; erreurs journalisées côté serveur ;
- modèle chargé une seule fois, prétraitement strictement identique
  entraînement/inférence ;
- exécution 100 % locale, sans service externe payant.

---

## 14. Références

1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant
   health to enable the development of mobile disease diagnostics*. arXiv:1511.08060.
2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for
   image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2:
   Inverted residuals and linear bottlenecks. *CVPR 2018*, 4510–4520.
4. Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE TKDE*, 22(10),
   1345–1359.
5. Chollet, F. (2021). *Deep Learning with Python* (2ᵉ éd.). Manning.

---

## 15. Licence

Code distribué sous licence MIT pour un usage académique. Les images du dataset
PlantVillage restent soumises aux conditions de leur source d'origine et ne sont pas
incluses dans ce dépôt.
