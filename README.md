# Détection et classification des maladies de la feuille de tomate

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
| Prétraitement | RGB, redimensionnement 224 × 224, normalisation [-1, 1] |
| Prédiction | MobileNetV2 + Transfer Learning, 4 classes, sortie softmax |
| Résultat | classe prédite, confiance, durée, probabilités des 4 classes |
| Confidentialité | aucune image ni aucun résultat conservé (traitement en mémoire) |

**Classes reconnues**

1. `Tomato___healthy` — feuille saine
2. `Tomato___Early_blight` — alternariose précoce (*Alternaria solani*)
3. `Tomato___Late_blight` — mildiou (*Phytophthora infestans*)
4. `Tomato___Leaf_Mold` — moisissure des feuilles (*Passalora fulva*)

---

## 2. Structure du dépôt

```text
plant-disease-detection/
├── app.py                      # application Flask : 2 routes (page + analyse)
├── config.py                   # chemins, hyperparamètres, libellés des classes
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/PlantVillage/       # images brutes (non versionnées)
│   ├── processed/              # train/validation/test + dataset_manifest.json
│   └── class_names.json        # ordre des classes du modèle
│
├── models/
│   ├── best_model.keras        # modèle final (livrable)
│   └── training_config.json    # hyperparamètres et résultats de l'entraînement
│
├── training/
│   ├── prepare_data.py         # contrôle qualité + split stratifié sans fuite
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
│   ├── figures/                # courbes, matrice, performances
│   ├── metrics/                # evaluation.json, classification_report.txt
│   └── predictions/            # prédictions du test, erreurs
│
├── tests/                      # pytest : prétraitement, utilitaires, application
│
├── scripts/                    # génération des livrables (rapport, fiche, slides)
└── docs/
    ├── rapport/                # rapport Markdown + Word + LaTeX/PDF
    ├── fiche/                  # fiche expérimentale
    └── presentation/           # diaporama PowerPoint
```

Chaque module a une responsabilité unique : `config.py` centralise chemins et
hyperparamètres, `utils/preprocessing.py` garantit un prétraitement identique entre
entraînement et inférence, `utils/predictor.py` charge le modèle une seule fois, et
`app.py` ne contient que les routes et le rendu.

---

## 3. Installation

Prérequis : Python 3.10 à 3.12 et `git` (pour le téléchargement du dataset).

```bash
git clone <url-du-depot> plant-disease-detection
cd plant-disease-detection

python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

> Sur une machine équipée d'un GPU NVIDIA, remplacer `tensorflow-cpu` par
> `tensorflow` dans `requirements.txt`.

---

## 4. Dataset

- **Nom** : PlantVillage Dataset (sous-ensemble tomate, images couleur)
- **Source** : <https://github.com/spMohanty/PlantVillage-Dataset>
- **Publication** : Hughes & Salathé (2015), arXiv:1511.08060 ; Mohanty, Hughes &
  Salathé (2016), *Frontiers in Plant Science*, 7, 1419.
- **Images** : 5 452 photographies couleur 256 × 256, 4 classes.

| Classe | Images |
| --- | --- |
| Tomato___Early_blight | 1 000 |
| Tomato___Late_blight | 1 909 |
| Tomato___Leaf_Mold | 952 |
| Tomato___healthy | 1 591 |
| **Total** | **5 452** |

Le dépôt d'origine ne contient pas de fichier de licence explicite : les images sont
utilisées dans un cadre strictement académique, ne sont pas redistribuées et ne sont pas
versionnées ici.

```bash
python training/prepare_data.py       # contrôle qualité + split 70/15/15
```

`prepare_data.py` vérifie chaque image, détecte les doublons exacts (MD5) et les place
dans le même ensemble, réalise un découpage **stratifié** avec la graine 42, puis écrit
`data/processed/dataset_manifest.json`.

Résultat attendu :

```text
Classe                         Train  Valid.    Test   Total
Tomato___Early_blight            700     150     150    1000
Tomato___Late_blight            1334     288     287    1909
Tomato___Leaf_Mold               666     143     143     952
Tomato___healthy                1111     240     240    1591
TOTAL                           3811     821     820    5452
```

---

## 5. Entraînement réduit (CPU)

```bash
python training/train.py --epochs 15 --fine-tune-epochs 8
```

Paramètres :

| Paramètre | Valeur |
| --- | --- |
| Taille d'entrée | 224 × 224 × 3 |
| Batch size | 32 |
| Phase 1 (base gelée) | 30 époques max, Early Stopping (patience 5), Checkpoint |
| Phase 2 (fine-tuning) | 15 époques max, 30 dernières couches dégelées, Adam 1e-5 |
| Poids de classes | activés (compensent le déséquilibre) |
| Graine | 42 |

Déroulement ((~ 35 min sur CPU à 2 cœurs) :

- **phase 1** : base gelée, Adam 1e-3, Early Stopping + Checkpoint sur la meilleure
  validation ;
- **phase 2** : on recharge ce meilleur modèle, on dégèle les 30 dernières couches
  (BatchNormalization exclues), on recompile à 1e-5 et on continue les époques.

**Pourquoi deux phases** : pendant la phase 1, la base (MobileNetV2 pré-entraînée sur
ImageNet) reste gelée, seule la petite tête apprise appris ; lors de la phase 2, on
relâche les couches les plus spécifiques pour les adapter au domaine des feuilles de
tomate sans détruire les filtres déjà utiles.

**Pourquoi pas de Data Augmentation imbriquée** : les transformations (rotation, zoom,
flip, luminosité, translation) restent dans le pipeline de chargement des données, pas
dans le modèle sauvegardé ; le prétraitement applique donc la même opération à chaque
image, que ce soit à l'entraînement ou à la prédiction (déjà identique à
224 × 224 → redimensionnement 224 × 224 → normalisation 224 × 224 → normalisation [-1,
1]).

---

## 6. Évaluation

```bash
python training/evaluate.py
```

Sorties : `results/metrics/evaluation.json`, `classification_report.txt`,
`evaluation_summary.md`, `results/figures/*.png`, `results/predictions/*`.

Toutes les métriques sont calculées à partir des prédictions réelles du modèle sur
l'ensemble de test. Aucune valeur n'est simulée ni recopiée d'une publication.

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

Le résultat affiche la classe prédite, la confiance, la durée de l'inférence et les
quatre probabilités. Aucune image n'est enregistrée : le fichier est traité en mémoire.
Il n'y a ni base de données, ni historique, ni API JSON, ni JavaScript.

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

33 tests couvrent :

- **app** : page d'accueil, analyse réelle de bout en bout (image → prédiction →
  affichage), refus des fichiers invalides, erreurs 400/404/413 ;
- **prétraitement** : conversion RGB, redimensionnement, normalisation, équivalence
  exacte avec `mobilenet_v2.preprocess_input` ;
- **utilitaires** : extensions autorisées, refus des faux types MIME, des contenus
  invalides, des images trop petites ou trop grandes ;
- **modèle** : chargement, prédire(first), prédiction, nommage des classes.

---

## 10. Résultats expérimentaux

Résultats mesurés sur l'ensemble de test indépendant (820 images) :

| Métrique (test, 820 images) | Valeur |
| --- | --- |
| Exactitude (accuracy) | **95,85 %** |
| Précision macro | 95,47 % |
| Rappel macro | 95,25 % |
| F1-score macro | **95,34 %** |
| F1-score pondéré | 95,82 % |
| Images mal classées | 34 (4,15 %) |
| Meilleure exactitude de validation | 95,98 % (époque 22) |

| Classe | Précision | Rappel | F1-score | Support |
| --- | --- | --- | --- | --- |
| Alternariose précoce | 93,66 % | 88,67 % | 91,10 % | 150 |
| Mildiou | 96,15 % | 95,82 % | 95,99 % | 287 |
| Moisissure des feuilles | 94,52 % | 96,50 % | 95,50 % | 143 |
| Feuille saine | 97,56 % | 100,00 % | 98,77 % | 240 |

Les 34 erreurs se concentrent sur la paire alternariose / mildiou (9 + 6 cas). La
confiance moyenne est de 97,52 % pour les prédictions correctes contre 70,23 % pour les
erreurs.

Ces résultats portent sur un entraînement réellement exécuté en deux phases, avec un
budget d'époques réduit (15 + 8) en raison d'un environnement sans GPU.

---

## 11. Reproductibilité et limites

**Reproductibilité** : graine fixée (42), découpage stratifié déterministe, détection des
doublons, hyperparamètres centralisés et journalisés.

**Limites principales** :

- quatre classes seulement, alors que PlantVillage en compte 38 ;
- images de laboratoire (fond uniforme) ; le modèle n'a jamais vu de conditions réelles ;
- déséquilibre entre classes (952 à 1 909 images) ;
- similarité visuelle entre mildiou et alternariose : principale source de confusion ;
- aucune indication sur la gravité ou le stade de la maladie ;
- ensemble de test issu de la même distribution que l'entraînement.

---

## 12. Livrables universitaires

| Livrable | Emplacement |
| --- | --- |
| Code source complet | ce dépôt |
| Modèle entraîné | `models/best_model.keras` + `models/training_config.json` |
| Résultats et figures | `results/` |
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

> Le rapport LaTeX conserve des **espaces réservés** (auteur, encadrant, établissement,
> année) : ils se complètent dans les quatre macros `\report...` en tête de
> `docs/rapport/latex/rapport_final.tex`. Les valeurs saisies sont conservées si le
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
