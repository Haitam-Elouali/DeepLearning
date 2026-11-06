# Détection et classification des maladies des plantes par CNN et Transfer Learning

**Application expérimentale de reconnaissance visuelle des maladies de la feuille de tomate.**

Projet universitaire de Deep Learning — TensorFlow/Keras, MobileNetV2, Transfer Learning,
Flask, SQLite, Bootstrap 5 et Chart.js. Interface entièrement en français.

> ⚠️ **Avertissement** : cet outil est un projet pédagogique d'aide à la reconnaissance
> visuelle. Il ne constitue pas un diagnostic agricole et ne remplace ni l'observation
> d'un agronome ni une analyse de laboratoire.

---

## 1. Fonctionnalités

| Fonctionnalité | Détail |
| --- | --- |
| Téléversement d'image | JPG, JPEG, PNG — 8 Mo maximum, validation du type réel |
| Prétraitement | Conversion RGB, redimensionnement 224 × 224, normalisation [-1, 1] |
| Prédiction | MobileNetV2 + Transfer Learning, 4 classes, sortie Softmax |
| Résultats | Classe prédite, confiance, probabilités des 4 classes, graphique |
| Fiches maladies | Description, symptômes, prévention, images d'exemple |
| Historique | Tableau des analyses, page de détail, suppression |
| Tableau de bord | Compteurs, répartition des prédictions, état du modèle |
| API JSON | `/api/analyser`, `/api/historique`, `/api/statistiques`, `/sante` |
| Hors-ligne | Bootstrap et Chart.js embarqués localement, aucune API payante |

**Classes reconnues**

1. `Tomato___healthy` — feuille saine
2. `Tomato___Early_blight` — alternariose précoce (*Alternaria solani*)
3. `Tomato___Late_blight` — mildiou (*Phytophthora infestans*)
4. `Tomato___Leaf_Mold` — moisissure des feuilles (*Passalora fulva*)

---

## 2. Architecture du projet

```text
plant-disease-detection/
├── app.py                      # Application Flask (pages + API JSON)
├── config.py                   # Configuration centrale (chemins, hyperparamètres)
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/PlantVillage/       # images brutes (non versionnées)
│   ├── processed/              # train/validation/test + dataset_manifest.json
│   ├── class_names.json        # ordre des classes du modèle
│   ├── disease_info.json       # fiches pédagogiques des 4 classes
│   └── history.db              # base SQLite (générée au premier lancement)
│
├── models/
│   ├── best_model.keras        # modèle final (livrable)
│   └── training_config.json    # hyperparamètres et résultats de l'entraînement
│
├── notebooks/
│   └── plant_disease_training.ipynb   # notebook d'expérimentation
│
├── training/
│   ├── download_dataset.py     # téléchargement PlantVillage (clone partiel)
│   ├── prepare_data.py         # contrôle qualité + split stratifié sans fuite
│   ├── train.py                # MobileNetV2 : phase 1 + fine-tuning
│   ├── evaluate.py             # métriques, matrice, figures, analyse d'erreurs
│   └── predict.py              # inférence en ligne de commande
│
├── utils/
│   ├── preprocessing.py        # pipeline d'images partagé train/inférence
│   ├── database.py             # couche SQLite (historique)
│   ├── predictor.py            # chargement unique du modèle + prédiction
│   └── helpers.py              # validation fichiers, métadonnées, formatage
│
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── vendor/                 # Bootstrap 5, Bootstrap Icons, Chart.js (local)
│   ├── images/examples/        # exemples extraits du dataset
│   └── uploads/                # images téléversées (non versionnées)
│
├── templates/                  # base, index, analyze, history, detail, diseases, about
│
├── results/
│   ├── figures/                # courbes, matrice de confusion, performances
│   ├── metrics/                # evaluation.json, classification_report.txt, CSV
│   └── predictions/            # prédictions du test, erreurs, planche d'erreurs
│
├── tests/                      # tests pytest (prétraitement, helpers, base, app)
│
├── scripts/
│   ├── build_docx.py           # convertisseur Markdown → Word (livrables)
│   ├── build_notebook.py       # génération du notebook
│   └── build_presentation.py   # génération de la présentation PowerPoint
│
└── docs/
    ├── rapport/                # rapport scientifique (Markdown + DOCX)
    ├── fiche/                  # fiche expérimentale
    └── presentation/           # diaporama PowerPoint
```

Chaque module a une responsabilité unique :

- `config.py` centralise **tous** les chemins et hyperparamètres ;
- `utils/preprocessing.py` garantit l'identité stricte du prétraitement entre
  entraînement et inférence (vérifiée par un test) ;
- `utils/predictor.py` charge le modèle **une seule fois** (singleton thread-safe) ;
- `utils/database.py` est la seule couche qui parle SQL ;
- `app.py` ne contient ni logique de modèle ni SQL.

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

### Origine et conditions d'utilisation

- **Nom** : PlantVillage Dataset (sous-ensemble tomate, images couleur)
- **Source** : <https://github.com/spMohanty/PlantVillage-Dataset>
- **Publication** : Hughes, D. P. & Salathé, M. (2015). *An open access repository of
  images on plant health to enable the development of mobile disease diagnostics*.
  arXiv:1511.08060.
- **Article de référence** : Mohanty, S. P., Hughes, D. P. & Salathé, M. (2016).
  *Using deep learning for image-based plant disease detection*. Frontiers in Plant
  Science, 7, 1419.
- **Images** : 5 452 photographies couleur de feuilles de tomate, 256 × 256 pixels,
  réparties en 4 classes.

Le dépôt GitHub officiel **ne contient pas de fichier de licence explicite**. Les
images sont diffusées à des fins de recherche ; elles sont donc utilisées ici dans un
cadre strictement académique, ne sont pas redistribuées et ne sont pas versionnées
dans ce dépôt. Vérifier les conditions d'usage auprès de la source avant toute
rediffusion.

| Classe | Images |
| --- | --- |
| Tomato___Early_blight | 1 000 |
| Tomato___Late_blight | 1 909 |
| Tomato___Leaf_Mold | 952 |
| Tomato___healthy | 1 591 |
| **Total** | **5 452** |

### Téléchargement

```bash
python training/download_dataset.py
```

Le script effectue un clone partiel du dépôt (`--filter=blob:none` + checkout creux)
et ne récupère que les quatre dossiers utiles (≈ 100 Mo au lieu de plusieurs Go).

### Préparation (split stratifié sans fuite)

```bash
python training/prepare_data.py
```

Le script :

1. vérifie chaque image (décodage Pillow, dimensions) et écarte les fichiers invalides ;
2. détecte les **doublons exacts** (empreinte MD5) et les place dans le même ensemble ;
3. réalise un découpage **stratifié 70 / 15 / 15** avec la graine 42 ;
4. écrit `data/processed/dataset_manifest.json` (comptages, ratios, contrôles) ;
5. copie 3 images d'exemple par classe dans `static/images/examples/`.

Résultat attendu :

```text
Classe                         Train  Valid.    Test   Total
Tomato___Early_blight            700     150     150    1000
Tomato___Late_blight            1334     288     287    1909
Tomato___Leaf_Mold               666     143     143     952
Tomato___healthy                1111     240     240    1591
TOTAL                           3811     821     820    5452
```

Option utile sur machine modeste : `--max-per-class 400` (sous-échantillon
documenté dans le manifeste).

---

## 5. Entraînement

```bash
python training/train.py
```

Paramètres par défaut (modifiables dans `config.py` ou en ligne de commande) :

| Paramètre | Valeur |
| --- | --- |
| Taille d'entrée | 224 × 224 × 3 |
| Batch size | 32 |
| Époques (phase 1) | 30 (arrêt anticipé, patience 5) |
| Époques (fine-tuning) | 15 |
| Optimiseur | Adam — 1e-3 puis 1e-5 |
| Perte | Categorical crossentropy |
| Graine | 42 |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger |

Deux phases :

1. **Extraction de caractéristiques** — base MobileNetV2 gelée, entraînement de la tête
   `GAP → Dropout → Dense(128) → Dropout → Dense(4, softmax)` ;
2. **Fine-tuning partiel** — dégel des 30 dernières couches (BatchNormalization exclu),
   learning rate 1e-5, sauvegarde du meilleur modèle par accuracy de validation.

Exemple d'exécution réduite (CPU lent) :

```bash
python training/train.py --epochs 15 --fine-tune-epochs 8
```

Artefacts produits : `models/best_model.keras`, `models/training_config.json`,
`results/metrics/training_history.json`, `results/metrics/training_log.csv`.

---

## 6. Évaluation

```bash
python training/evaluate.py
```

Sorties :

- `results/metrics/evaluation.json` — accuracy, precision, recall, F1 (macro et
  pondérés), métriques par classe, matrice de confusion, statistiques de confiance ;
- `results/metrics/classification_report.txt` — rapport scikit-learn complet ;
- `results/figures/training_curves.png` — accuracy et loss par époque ;
- `results/figures/confusion_matrix.png` — effectifs et proportions ;
- `results/figures/per_class_performance.png` — precision / recall / F1 par classe ;
- `results/figures/confidence_distribution.png` — confiance succès vs erreurs ;
- `results/predictions/predictions_test.csv` et `erreurs.csv` ;
- `results/predictions/error_grid.png` — planche d'images mal classées.

> Toutes les métriques sont calculées à partir des prédictions réelles du modèle sur
> l'ensemble de test. Aucune valeur n'est simulée ni recopiée d'une publication.

---

## 7. Application Web

```bash
python app.py
# ou
flask --app app run --debug
```

Ouvrir <http://127.0.0.1:5000>.

| Page | Route | Contenu |
| --- | --- | --- |
| Tableau de bord | `/` | compteurs, classes, répartition, dernières analyses |
| Analyse | `/analyse` | dépôt d'image, aperçu, résultat, graphique |
| Détail d'une analyse | `/analyse/<id>` | image, classe, confiance, probabilités |
| Historique | `/historique` | tableau, suppression, accès aux détails |
| Maladies | `/maladies` | fiches des 4 classes |
| À propos | `/a-propos` | contexte, méthodologie, résultats, limites |

Le modèle est chargé **au démarrage** de l'application ; si le fichier
`models/best_model.keras` est absent, l'interface reste consultable et l'API renvoie
une erreur explicite (503) au lieu de planter.

### Variables d'environnement (optionnelles)

| Variable | Rôle | Défaut |
| --- | --- | --- |
| `PLANT_SECRET_KEY` | clé de session Flask | valeur de développement |
| `PLANT_DB_PATH` | chemin de la base SQLite | `data/history.db` |

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

Couverture fonctionnelle :

- **prétraitement** : conversion RGB, redimensionnement, normalisation, équivalence
  exacte avec `mobilenet_v2.preprocess_input` ;
- **helpers** : extensions autorisées, assainissement des noms, refus des faux types
  MIME, des contenus invalides et des images trop petites ;
- **base de données** : ajout, lecture, tri, suppression, statistiques ;
- **application** : pages, sonde `/sante`, erreurs 400/404/413, parcours complet
  téléversement → prédiction réelle → historique → suppression, refus d'un nom de
  fichier malveillant.

Les tests qui utilisent le modèle et les images de test sont automatiquement ignorés
(`skip`) si le dataset ou le modèle n'est pas présent.

---

## 10. Résultats expérimentaux

Les résultats complets, mesurés sur l'ensemble de test indépendant (820 images),
sont disponibles dans [`results/metrics/evaluation.json`](results/metrics/evaluation.json),
dans [`results/metrics/evaluation_summary.md`](results/metrics/evaluation_summary.md)
et dans le rapport (`docs/rapport/rapport_final.docx`).

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
| Alternariose précoce (Early blight) | 93,66 % | 88,67 % | 91,10 % | 150 |
| Mildiou (Late blight) | 96,15 % | 95,82 % | 95,99 % | 287 |
| Moisissure des feuilles (Leaf Mold) | 94,52 % | 96,50 % | 95,50 % | 143 |
| Feuille saine | 97,56 % | 100,00 % | 98,77 % | 240 |

Les 34 erreurs se concentrent sur la paire alternariose / mildiou (9 + 6 cas),
deux maladies dont les lésions se ressemblent sur les images de PlantVillage. La
confiance moyenne est de 97,52 % pour les prédictions correctes contre 70,23 %
pour les erreurs : le modèle « sait » en partie lorsqu'il hésite.

Ces résultats portent sur un entraînement réellement exécuté en deux phases, avec un budget
d'époques réduit (15 + 8) en raison d'un environnement d'exécution sans GPU
(processeur à 2 cœurs). La configuration permet 30 + 15 époques.

---

## 11. Reproductibilité et limites

**Reproductibilité** : graine fixée (42), découpage stratifié déterministe,
détection des doublons, hyperparamètres centralisés et journalisés.

**Limites principales** :

- quatre classes seulement, alors que PlantVillage en compte 38 ;
- images de laboratoire (fond uniforme) ; l'écart avec des photos de terrain est
  important, le modèle n'a jamais vu de conditions réelles ;
- déséquilibre entre classes (952 à 1 909 images) ;
- similarité visuelle entre mildiou et alternariose : principale source de confusion ;
- aucune indication sur la gravité ou le stade de la maladie ;
- ensemble de test issu de la même distribution que l'entraînement.

**Perspectives** : ajouter des classes, tester sur des photos de terrain, validation
croisée, Grad-CAM pour l'interprétabilité, quantification du modèle pour un usage
mobile.

---

## 12. Livrables universitaires

| Livrable | Emplacement |
| --- | --- |
| Code source complet | ce dépôt |
| Notebook d'expérimentation | `notebooks/plant_disease_training.ipynb` |
| Data Augmentation | `training/train.py` (`construire_augmentation`) |
| Modèle entraîné | `models/best_model.keras` + `models/training_config.json` |
| Résultats et figures | `results/` |
| Rapport scientifique | `docs/rapport/rapport_final.docx` |
| Fiche expérimentale | `docs/fiche/fiche_experimentale.docx` |
| Présentation orale | `docs/presentation/presentation_soutenance.pptx` |

Régénération des livrables :

```bash
python scripts/build_report.py          # rapport Markdown + Word
python scripts/build_fiche.py           # fiche expérimentale
python scripts/build_presentation.py    # diaporama PowerPoint
```

---

## 13. Sécurité et qualité

- extensions et types MIME contrôlés, contenu réellement décodé par Pillow ;
- taille maximale (8 Mo) et dimensions maximales (4 000 px) ;
- ré-encodage systématique des images en JPEG RGB et renommage par UUID ;
- service des images via une route validant un motif strict (aucune traversée de
  répertoire) ;
- aucun chemin interne renvoyé par l'API ; erreurs journalisées côté serveur ;
- requêtes SQL paramétrées (aucune concaténation) ;
- modèle chargé une seule fois, préprocessing strictement identique
  entraînement/inférence ;
- exécution 100 % locale, sans service externe payant.

---

## 14. Références

1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant
   health to enable the development of mobile disease diagnostics*. arXiv:1511.08060.
2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for
   image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
   <https://doi.org/10.3389/fpls.2016.01419>
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2:
   Inverted residuals and linear bottlenecks. *CVPR 2018*, 4510–4520.
4. Howard, A. G. et al. (2017). *MobileNets: Efficient convolutional neural networks
   for mobile vision applications*. arXiv:1704.04861.
5. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
6. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with
   deep convolutional neural networks. *NeurIPS 2012*.
7. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
   recognition. *CVPR 2016*.
8. Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for
   deep learning. *Journal of Big Data*, 6(1), 60.
9. Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on
   Knowledge and Data Engineering*, 22(10), 1345–1359.
10. Chollet, F. (2021). *Deep Learning with Python* (2ᵉ éd.). Manning.

---

## 15. Licence

Code distribué sous licence MIT pour un usage académique. Les images du dataset
PlantVillage restent soumises aux conditions de leur source d'origine et ne sont pas
incluses dans ce dépôt.
