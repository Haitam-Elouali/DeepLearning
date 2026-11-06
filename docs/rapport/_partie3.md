## 11. Développement de l'application Web

### 11.1 Objectifs et architecture générale

L'application Web constitue la couche de restitution du projet : elle doit rendre le
modèle utilisable par une personne qui ne connaît rien à l'apprentissage automatique.
Elle a été conçue selon une architecture en trois couches :

1. **couche de présentation** : templates Jinja2, HTML5, CSS3, Bootstrap 5, JavaScript
   natif et Chart.js ;
2. **couche applicative** : Flask, validation des entrées, routage, API JSON, gestion des
   erreurs ;
3. **couche métier** : prédicteur Keras (chargé une seule fois), base SQLite pour
   l'historique, utilitaires de prétraitement.

Chaque responsabilité est isolée dans un module distinct, ce qui facilite la maintenance
et les tests :

| Module | Responsabilité |
| --- | --- |
| `config.py` | chemins, hyperparamètres, paramètres de sécurité |
| `app.py` | routes, API, gestion des erreurs, rendu des templates |
| `utils/preprocessing.py` | pipeline d'image partagé entraînement / inférence |
| `utils/predictor.py` | chargement unique du modèle et prédiction |
| `utils/database.py` | accès SQLite (une seule couche qui parle SQL) |
| `utils/helpers.py` | validation des fichiers, métadonnées, formatage |

*Tableau 11 — Modules du projet et responsabilités.*

### 11.2 Pages et fonctionnalités

| Page | Route | Contenu principal |
| --- | --- | --- |
| Tableau de bord | `/` | compteurs, état du modèle, classes, répartition, dernières analyses |
| Analyse | `/analyse` | dépôt d'image, aperçu, prédiction, probabilités, description |
| Détail d'analyse | `/analyse/<id>` | image, classe, confiance, probabilités, symptômes |
| Historique | `/historique` | tableau des analyses, suppression, accès aux détails |
| Maladies | `/maladies` | fiches des quatre classes avec exemples et prévention |
| À propos | `/a-propos` | contexte, méthodologie, architecture, limites, références |

*Tableau 12 — Pages de l'application et routes associées.*

**Tableau de bord.** Il affiche le nombre total d'analyses, la confiance moyenne, les
analyses de la semaine, le nombre de classes, l'état de chargement du modèle, la
répartition des prédictions sous forme de graphique en anneau et les cinq dernières
analyses. C'est la page d'accueil : elle doit donner en un coup d'œil une idée du
fonctionnement et de l'activité du système.

![Tableau de bord de l'application](captures/01_tableau_de_bord.png)

*Figure 5 — Tableau de bord : présentation du projet et état du système (affichage mobile).*

![Statistiques et répartition des prédictions](captures/02_statistiques_classes.png)

*Figure 6 — Tableau de bord : répartition des classes prédites sous forme de graphique.*

**Page d'analyse.** L'utilisateur dépose une image par glisser-déposer ou par sélection
de fichier. Un aperçu s'affiche immédiatement, avant toute analyse, accompagné du nom et
de la taille du fichier. Le bouton « Lancer l'analyse » déclenche un appel asynchrone
(`fetch`) vers `/api/analyser`, sans rechargement de page, avec un indicateur de
chargement (spinner) pendant l'inférence.

![Page d'analyse avec aperçu de l'image](captures/05_apercu_image.png)

*Figure 7 — Page d'analyse : aperçu de l'image avant lancement de la prédiction.*

Le résultat présente la classe prédite, un résumé de la maladie, le pourcentage de
confiance, la durée de l'inférence, un graphique en barres des quatre probabilités, la
liste détaillée des scores, la description de la maladie et ses symptômes
caractéristiques. L'utilisateur peut relancer une analyse ou consulter la fiche
complète de l'analyse enregistrée.

![Résultat d'une analyse](captures/07_resultat_entete.png)

*Figure 8 — Résultat d'une analyse : classe prédite, confiance et distribution des
probabilités.*

**Historique.** Les analyses sont conservées dans une base SQLite et affichées dans un
tableau trié par date décroissante : identifiant, date et heure, nom de l'image, maladie
prédite, score de confiance, accès au détail et suppression. La suppression est protégée
par une fenêtre de confirmation et efface également le fichier image associé.

![Historique des analyses](captures/08_historique.png)

*Figure 9 — Historique des analyses enregistrées dans la base SQLite.*

**Fiches maladies.** Chaque classe est présentée avec trois images d'exemple issues du
dataset, une description, les symptômes visuels caractéristiques, des conseils de
prévention et une mention des confusions possibles. Cette page rappelle explicitement que
le périmètre est limité à quatre catégories.

![Fiche descriptive d'une maladie](captures/10_maladies.png)

*Figure 10 — Fiche maladie : description, symptômes et prévention.*

**À propos.** Cette page documente le contexte scientifique, la problématique, les
objectifs, l'architecture, les technologies, la configuration réelle du modèle et ses
limites. Elle contribue à la posture honnête du projet.

![Page À propos](captures/11_a_propos.png)

*Figure 11 — Page « À propos » : contexte, objectifs et méthodologie.*

### 11.3 API JSON

L'application expose une API simple, utilisée par le frontend et exploitable par un
script :

| Méthode | Route | Rôle |
| --- | --- | --- |
| POST | `/api/analyser` | téléversement + prédiction + enregistrement |
| GET | `/api/historique` | liste des analyses au format JSON |
| DELETE | `/api/historique/<id>` | suppression d'une analyse et de son image |
| GET | `/api/statistiques` | statistiques agrégées |
| GET | `/sante` | sonde de disponibilité (modèle chargé, nombre de classes) |

*Tableau 13 — Points d'entrée de l'API JSON.*

La réponse de `/api/analyser` contient la classe, le nom français, la confiance, les
quatre probabilités triées, la durée d'inférence, l'URL de l'image et le lien vers la
fiche détaillée. En cas d'erreur, un code HTTP approprié (400, 404, 413, 500, 503) est
renvoyé avec un message compréhensible en français.

### 11.4 Base de données

L'historique est stocké dans une base SQLite unique, créée automatiquement au premier
lancement. Le schéma est volontairement minimal :

| Colonne | Type | Rôle |
| --- | --- | --- |
| `id` | INTEGER (PK) | identifiant de l'analyse |
| `nom_stocke` | TEXT | nom aléatoire du fichier sur le serveur (UUID) |
| `nom_original` | TEXT | nom d'origine assaini, pour l'affichage |
| `classe_predite` | TEXT | classe retournée par le modèle |
| `confiance` | REAL | probabilité de la classe prédite (0 à 1) |
| `probabilites` | TEXT | distribution complète au format JSON |
| `modele` | TEXT | nom du fichier modèle utilisé |
| `duree_ms` | INTEGER | durée de l'inférence en millisecondes |
| `created_at` | TEXT | horodatage ISO 8601 |

*Tableau 14 — Schéma de la table `analyses`.*

Toutes les requêtes utilisent des paramètres liés (`?`), ce qui élimine les risques
d'injection SQL. Aucune donnée personnelle n'est stockée : ni adresse IP, ni identifiant
utilisateur.

### 11.5 Sécurité et robustesse

Les mesures suivantes ont été mises en œuvre :

- **validation des extensions** contre une liste blanche (`.jpg`, `.jpeg`, `.png`) ;
- **vérification du type MIME** déclaré par le navigateur ;
- **décodage réel du fichier** par Pillow : une image corrompue ou un fichier renommé
  est rejeté, l'extension seule ne suffit pas ;
- **taille maximale** de 8 Mo, appliquée par Flask (`MAX_CONTENT_LENGTH`) et par le
  frontend ;
- **dimensions maximales** de 4 000 pixels et taille minimale de 32 pixels ;
- **renommage systématique** par UUID : aucun nom fourni par l'utilisateur n'est utilisé
  pour construire un chemin ;
- **ré-encodage en JPEG RGB** : neutralise les métadonnées et les charges utiles cachées ;
- **service des images** via une route validant un motif strict
  (`^[0-9a-f]{32}\.jpg$`), ce qui empêche toute traversée de répertoire ;
- **aucun chemin interne** renvoyé par l'API ; les erreurs techniques sont journalisées
  côté serveur et remplacées par un message générique ;
- **chargement unique du modèle** au démarrage (singleton protégé par un verrou), et non
  à chaque requête ;
- **gestion explicite** des erreurs 404, 413, 500 et du cas « modèle absent » (503).

### 11.6 Choix d'interface et accessibilité

L'interface utilise une palette verte et rouge évoquant le végétal et la tomate, des
cartes arrondies, des transitions discrètes et une typographie lisible. Les éléments
interactifs possèdent un état de focus visible, la zone de dépôt est utilisable au clavier
(`tabindex` et gestion des touches Entrée/Espace), les images ont un texte alternatif et
les graphiques portent un label ARIA. Bootstrap 5 assure la responsivité : sur mobile, la
navigation se replie et les cartes passent sur une colonne.

Bootstrap, Bootstrap Icons et Chart.js sont **embarqués localement** dans
`static/vendor/`, ce qui permet une démonstration hors ligne, sans dépendance à un CDN.

### 11.7 Tests de l'application

Quarante-quatre tests automatisés (pytest) couvrent le prétraitement, les utilitaires, la
base de données et l'application. Ils incluent un parcours complet réel :
téléversement d'une image du jeu de test → prédiction par le vrai modèle → enregistrement
→ consultation de l'historique et du détail → suppression → vérification de la
disparition en base et sur le disque. Les tests vérifient également le refus d'un fichier
non-image, d'une extension interdite, d'un dépassement de taille et d'un nom de fichier
malveillant.

## 12. Résultats expérimentaux

### 12.1 Protocole

L'évaluation est réalisée par `training/evaluate.py` sur les **820 images de l'ensemble
de test**, jamais utilisées pendant l'entraînement ni pendant la sélection du modèle. Le
script charge `models/best_model.keras`, applique exactement le prétraitement décrit en
section 8, calcule les prédictions par lots, puis dérive l'ensemble des métriques avec
scikit-learn. Les figures et les fichiers CSV sont générés automatiquement.

### 12.2 Courbes d'apprentissage

![Courbes d'accuracy et de perte](../../results/figures/training_curves.png)

*Figure 12 — Courbes d'apprentissage : accuracy et perte, entraînement et validation,
avec la séparation entre phase 1 et fine-tuning (ligne pointillée).*

L'accuracy d'entraînement progresse régulièrement de 77,7 % à 96,7 %. L'accuracy de
validation atteint rapidement 94,5 %, puis le fine-tuning permet de franchir la barre des
95 % à l'époque 22 (**95,98 %**). La perte de validation descend de 0,33 à 0,12. L'écart
final entre entraînement (96,7 %) et validation (95,98 %) reste modéré, ce qui indique
un sur-apprentissage limité.

### 12.3 Métriques globales sur le test

| Métrique | Valeur |
| --- | --- |
| Accuracy | **95,85 %** |
| Precision (macro) | 95,47 % |
| Recall (macro) | 95,25 % |
| F1-score (macro) | **95,34 %** |
| Precision (pondérée) | 95,83 % |
| Recall (pondéré) | 95,85 % |
| F1-score (pondéré) | 95,82 % |
| Images de test | 820 |
| Images mal classées | 34 (4,15 %) |

*Tableau 15 — Métriques globales du modèle sur l'ensemble de test.*

L'accuracy de 95,85 % signifie que 786 images sur 820 ont été correctement classées. La
faible différence entre les moyennes macro (95,34 %) et pondérée (95,82 %) indique que
les performances sont relativement homogènes entre classes : le modèle n'a pas sacrifié
les classes minoritaires au profit des classes majoritaires.

### 12.4 Performances par classe

| Classe | Precision | Recall | F1-score | Support | Bien classées |
| --- | --- | --- | --- | --- | --- |
| Alternariose précoce | 93,66 % | 88,67 % | 91,10 % | 150 | 133 |
| Mildiou de la tomate | 96,15 % | 95,82 % | 95,99 % | 287 | 275 |
| Moisissure des feuilles | 94,52 % | 96,50 % | 95,50 % | 143 | 138 |
| Feuille saine | 97,56 % | 100,00 % | 98,77 % | 240 | 240 |
| **Macro** | **95,47 %** | **95,25 %** | **95,34 %** | 820 | 786 |

*Tableau 16 — Métriques détaillées par classe.*

![Performances par classe](../../results/figures/per_class_performance.png)

*Figure 13 — Precision, recall et F1-score pour chacune des quatre classes.*

Trois observations se dégagent :

1. **La classe « feuille saine » est parfaitement détectée** (rappel de 100 %). Aucune
   feuille saine n'a été classée comme malade, ce qui est le comportement souhaité pour un
   outil d'alerte : pas de fausse alerte sur un feuillage sain.
2. **L'alternariose précoce est la classe la plus difficile** (F1 = 91,10 %, rappel de
   88,67 %) : 17 feuilles sur 150 lui échappent, dont 9 classées comme mildiou. Ces deux
   maladies produisent des taches brunes et leur ressemblance visuelle est bien connue en
   agronomie.
3. **Le mildiou et la moisissure** obtiennent des scores élevés (F1 ≈ 96 %), cohérents
   avec leurs signatures visuelles plus marquées (lésions humides, duvet olive).

### 12.5 Matrice de confusion

| Réel \\ Prédit | Alternariose | Mildiou | Moisissure | Saine |
| --- | --- | --- | --- | --- |
| **Alternariose** | **133** | 9 | 4 | 4 |
| **Mildiou** | 6 | **275** | 4 | 2 |
| **Moisissure** | 3 | 2 | **138** | 0 |
| **Saine** | 0 | 0 | 0 | **240** |

*Tableau 17 — Matrice de confusion (effectifs) sur les 820 images de test.*

![Matrice de confusion](../../results/figures/confusion_matrix.png)

*Figure 14 — Matrice de confusion en effectifs (à gauche) et en proportions par classe
réelle (à droite).*

La lecture par ligne est la plus instructive : 88,7 % des alternarioses sont correctement
identifiées, 95,8 % des mildious, 96,5 % des moisissures et 100 % des feuilles saines. La
diagonale est nettement dominante, ce qui traduit un modèle globalement fiable. Les
erreurs se concentrent sur la frontière alternariose / mildiou, avec 15 confusions croisées
(9 + 6), soit près de la moitié des 34 erreurs totales.

### 12.6 Analyse des erreurs

| Indicateur | Valeur |
| --- | --- |
| Erreurs totales | 34 / 820 (4,15 %) |
| Confiance moyenne sur les prédictions correctes | 97,52 % |
| Confiance moyenne sur les erreurs | 70,23 % |
| Confiance médiane globale | 99,95 % |
| Confusion la plus fréquente | Alternariose → Mildiou (9 cas) |

*Tableau 18 — Statistiques de confiance et principales confusions.*

La confiance moyenne des erreurs (70,2 %) est très inférieure à celle des prédictions
correctes (97,5 %) : le modèle « sait » en partie qu'il hésite. C'est un signal utile pour
l'utilisateur et un argument en faveur de l'affichage de la distribution complète des
probabilités plutôt que d'une classe unique.

![Distribution de la confiance](../../results/figures/confidence_distribution.png)

*Figure 15 — Distribution de la confiance selon que la prédiction est correcte ou non.*

![Planche d'erreurs de classification](../../results/predictions/error_grid.png)

*Figure 16 — Exemples d'images mal classées : classe réelle, classe prédite et confiance.*

La planche d'erreurs montre que les images mal classées sont souvent des cas ambigus :
lésions peu contrastées, feuille partiellement floue, ou symptômes mélangés. Certaines
images annotées « alternariose précoce » présentent des taches sombres étendues qui
évoquent visuellement le mildiou. Ces erreurs ne relèvent donc pas uniquement d'une
faiblesse du modèle, mais aussi de la frontière floue entre deux pathologies.

### 12.7 Temps d'inférence

| Contexte | Durée observée |
| --- | --- |
| Première requête (chargement + échauffement) | ≈ 880 ms |
| Requêtes suivantes (modèle en mémoire) | 68 – 80 ms |
| Batch d'évaluation (16 images) | ≈ 1,2 s, soit ≈ 75 ms par image |

*Tableau 19 — Temps d'inférence mesurés sur le processeur de test.*

Le chargement unique du modèle au démarrage permet de répondre en moins de 100 ms par
image, ce qui rend l'expérience fluide dans le navigateur. La première requête est plus
lente en raison de l'initialisation des graphes TensorFlow.

## 13. Évaluation et interprétation des métriques

### 13.1 Lecture critique de l'accuracy

Une accuracy de 95,85 % sur un jeu de test indépendant est un résultat solide pour un
projet réalisé sans GPU et avec un budget d'époques réduit. Il faut néanmoins rappeler
qu'un tel chiffre n'a de sens que replacé dans son contexte :

- le test provient de la **même distribution** que l'entraînement (même dataset, même
  protocole de prise de vue), ce qui facilite la tâche ;
- la classe majoritaire représente 35 % du test ; un classifieur naïf atteindrait déjà
  environ 35 % d'accuracy, et un modèle prédisant toujours « mildiou » obtiendrait 35 % ;
- les 820 images de test sont peu nombreuses à l'échelle des variations possibles d'une
  feuille de tomate malade.

L'accuracy est donc un indicateur de faisabilité, pas une garantie de performance sur le
terrain.

### 13.2 Cohérence entre les métriques

Les quatre métriques macro sont comprises entre 95,25 % et 95,47 %, avec un F1 de
95,34 %. Cette homogénéité signifie que le modèle n'est pas seulement bon sur les classes
majoritaires : il conserve un bon rappel sur la moisissure (143 images seulement) et sur
l'alternariose (150 images). Les poids de classes et la Data Augmentation ont
probablement contribué à cet équilibre.

La precision (95,47 %) est légèrement supérieure au recall (95,25 %) : le modèle commet
un peu plus de faux négatifs que de faux positifs. Dans un contexte de détection de
maladie, ce compromis est discutable : manquer une feuille malade (faux négatif) peut
être plus coûteux que signaler à tort une feuille saine. Un ajustement du seuil de
décision, ou une pondération plus forte des classes malades, pourrait être envisagé.

### 13.3 Interprétation des confusions

Les confusions observées sont **biologiquement cohérentes** :

- **alternariose → mildiou (9 cas)** : les deux maladies produisent des taches brunes
  irrégulières ; à un stade avancé, l'alternariose perd ses anneaux concentriques et
  ressemble au mildiou ;
- **mildiou → alternariose (6 cas)** : symétrique du cas précédent, souvent sur des
  lésions jeunes ou des images peu contrastées ;
- **alternariose → feuille saine (4 cas)** : taches rares ou très petites, insuffisantes
  pour déclencher la classe malade ;
- **mildiou → moisissure (4 cas)** et **alternariose → moisissure (4 cas)** : images
  présentant des plages jaunies communes aux deux pathologies.

Aucune confusion n'implique la classe « feuille saine » en sortie : le modèle ne prend
jamais une feuille malade pour une feuille saine de manière significative (2 cas de
mildiou classés sains au maximum). C'est un point rassurant pour un usage d'alerte.

### 13.4 Calibration et sur-confiance

La confiance médiane (99,95 %) est très élevée, et 100 % des feuilles saines du test sont
prédites avec une confiance proche de 1. Les réseaux de neurones modernes sont connus pour
être **mal calibrés** : leurs probabilités softmax sont souvent trop tranchées. Un score
de 100 % ne signifie donc pas une certitude absolue, mais une forte activation du modèle.

L'écart de confiance entre prédictions correctes (97,5 %) et erreurs (70,2 %) montre que
la confiance reste informative. Une perspective d'amélioration serait d'appliquer une
**température scaling** ou une calibration de Platt sur l'ensemble de validation, afin de
produire des probabilités plus réalistes.

### 13.5 Positionnement par rapport à la littérature

Mohanty et al. (2016) rapportent des accuracies supérieures à 99 % sur PlantVillage avec
des architectures comme AlexNet et GoogLeNet, sur l'ensemble des 38 classes et avec un
protocole d'entraînement bien plus long. Nos 95,85 % sur 4 classes ne sont pas directement
comparables : le périmètre est plus restreint, le budget d'entraînement beaucoup plus
faible (23 époques, CPU) et le découpage différent. Le résultat confirme néanmoins la
tendance de fond de la littérature : sur PlantVillage, les CNN pré-entraînés atteignent
une précision élevée avec un effort d'entraînement modéré.

## 14. Discussion des résultats

### 14.1 Apport du Transfer Learning

La progression observée valide l'intérêt du Transfer Learning :

| Étape | Accuracy de validation |
| --- | --- |
| Époque 1 (base gelée) | 86,24 % |
| Meilleure phase 1 (époque 4) | 94,52 % |
| Meilleure phase 2 (époque 22) | **95,98 %** |

*Tableau 20 — Progression de la validation au fil de l'expérience.*

Atteindre 86 % dès la première époque, avec seulement 164 484 paramètres entraînables,
aurait été impossible avec un réseau initialisé aléatoirement. Le gain du fine-tuning
(+1,46 point) est plus modeste mais réel : il montre que les filtres profonds de
MobileNetV2, initialement spécialisés sur ImageNet, bénéficient d'un ajustement aux
textures de feuilles.

### 14.2 Variabilité et sensibilité aux hyperparamètres

La courbe de validation est irrégulière (oscillations de ±2 points entre époques
voisines). Trois facteurs expliquent cette variabilité :

1. la taille limitée du jeu de validation (821 images, soit environ 26 lots de 32) ;
2. l'application aléatoire de la Data Augmentation, qui modifie la difficulté des lots ;
3. le déséquilibre des classes, partiellement compensé par les poids de classes.

Un usage plus stable serait obtenu avec une validation croisée à k plis, plus coûteuse en
temps de calcul mais beaucoup plus robuste pour estimer la performance réelle.

### 14.3 Coût de calcul et pertinence de MobileNetV2

Le modèle complet ne pèse que 2,4 millions de paramètres et environ 24 Mo sur disque. Une
époque d'entraînement dure 90 secondes sur deux cœurs, et une prédiction 70 ms. Ces
chiffres confirment que MobileNetV2 est un choix adapté à un déploiement local, sur un
ordinateur portable ou un mini-serveur, sans GPU ni service cloud. Un modèle plus profond
(ResNet-50, EfficientNet-B3) aurait probablement apporté quelques points supplémentaires
mais au prix d'un temps d'entraînement multiplié par 5 à 10 dans notre environnement.

### 14.4 Qualité des données versus complexité du modèle

Le principal facteur limitant n'est pas la capacité du modèle mais la **nature des
données**. Les images PlantVillage sont prises en laboratoire, sur fond uniforme, avec une
feuille unique et un éclairage contrôlé. Sur des photographies de terrain (feuillage
dense, arrière-plan complexe, ombres, reflets, flou de bougé), les performances chuteraient
très probablement. Améliorer le dataset (images de terrain, plusieurs feuilles par image,
conditions variées) serait plus rentable que complexifier l'architecture.

### 14.5 Enseignements méthodologiques

Ce projet illustre plusieurs principes importants :

- **l'identité du prétraitement** entre entraînement et inférence est critique : elle est
  ici garantie par un module unique et vérifiée par un test ;
- **la prévention des fuites** (doublons, split) conditionne la validité des métriques ;
- **le suivi de la validation** (Early Stopping, checkpoint) évite de conserver un modèle
  dégradé : sans `initial_value_threshold`, le modèle final aurait été moins bon que
  celui de la phase 1 ;
- **l'analyse des erreurs** apporte plus d'information que l'accuracy globale : elle
  révèle les frontières ambiguës entre maladies.

## 15. Limites et perspectives d'amélioration

### 15.1 Limites du système actuel

**Limites liées aux données**

- quatre classes seulement, alors que PlantVillage en compte 38 et que la tomate connaît
  une dizaine de maladies foliaires courantes ;
- images de laboratoire, non représentatives des conditions de terrain ;
- déséquilibre entre classes (952 à 1 909 images) ;
- absence de regroupement par feuille (*leaf grouping*), qui peut laisser passer des
  quasi-doublons entre ensembles ;
- jeu de test de 820 images, insuffisant pour une estimation à ±0,5 point près.

**Limites liées au modèle**

- une seule architecture et un seul protocole testés, faute de temps de calcul ;
- pas de validation croisée ;
- probabilités mal calibrées (sur-confiance) ;
- pas d'explicabilité intégrée : l'utilisateur ne voit pas quelles zones de la feuille ont
  motivé la décision ;
- pas de détection d'image hors domaine (une photo de fleur serait classée dans une des
  quatre catégories).

**Limites liées à l'application**

- pas d'authentification ni de comptes utilisateurs : l'historique est partagé ;
- pas de déploiement en production (serveur de développement Flask) ;
- base SQLite adaptée à un usage local, non à un usage multi-utilisateurs intensif ;
- interface en français uniquement.

### 15.2 Perspectives techniques

1. **Explicabilité (Grad-CAM)** : produire une carte de chaleur des zones discriminantes,
   superposée à l'image, afin de renforcer la confiance de l'utilisateur et de détecter
   les décisions fondées sur des artefacts.
2. **Détection d'inconnu** : ajouter un seuil de rejet ou une classe « autre » entraînée
   sur des images hors domaine, pour éviter de classer n'importe quelle image.
3. **Calibration des probabilités** : température scaling sur l'ensemble de validation.
4. **Validation croisée** à 5 plis pour une estimation plus fiable des performances.
5. **Comparaison d'architectures** : MobileNetV3, EfficientNet-B0, ResNet-50, avec le même
   protocole, pour objectiver le compromis précision / coût.
6. **Quantification et export** : conversion du modèle en TensorFlow Lite (INT8) ou ONNX
   pour un déploiement mobile.
7. **Augmentation plus agressive et adaptée** : mélange d'images (*MixUp*, *CutMix*),
   simulation d'ombres et de fonds variés, pour préparer le passage au terrain.
8. **Suivi d'expériences** : intégration de MLflow ou de TensorBoard pour comparer
   systématiquement les runs.

### 15.3 Perspectives applicatives

- ajout d'un compte utilisateur et d'un historique personnel ;
- export PDF d'un rapport d'analyse ;
- prise de photo directe depuis le smartphone (API `getUserMedia`) ;
- mode hors-ligne complet avec un modèle quantifié exécuté dans le navigateur
  (TensorFlow.js) ;
- extension à d'autres cultures (pomme de terre, vigne, pommier) ;
- interface multilingue.

## 16. Conclusion générale

Ce projet avait pour objectif de concevoir un système complet de détection et de
classification des maladies de la feuille de tomate, depuis la donnée brute jusqu'à une
application Web utilisable. Cet objectif a été atteint : un modèle MobileNetV2 adapté par
Transfer Learning a été entraîné sur 3 811 images, évalué sur 820 images indépendantes, et
intégré dans une application Flask en français.

Les résultats expérimentaux sont les suivants :

- **accuracy de 95,85 %** sur l'ensemble de test ;
- **F1-score macro de 95,34 %**, avec des performances homogènes entre les quatre classes
  (de 91,10 % à 98,77 %) ;
- **34 erreurs sur 820 images**, concentrées sur la frontière entre alternariose précoce
  et mildiou, deux maladies visuellement proches ;
- **temps d'inférence d'environ 70 ms** par image, compatible avec une utilisation
  interactive ;
- un modèle compact de **2,4 millions de paramètres**, entraînable et exécutable sans GPU.

Au-delà des chiffres, le projet apporte plusieurs contributions méthodologiques : un
découpage rigoureux sans fuite de données, un prétraitement strictement identique entre
entraînement et inférence (vérifié par un test), une évaluation complète avec matrice de
confusion et analyse d'erreurs, une application sécurisée et testée, et une documentation
qui distingue clairement ce qui a été mesuré de ce qui reste hypothétique.

Le système ne constitue pas un diagnostic agricole. Il s'agit d'un outil expérimental
d'aide à la reconnaissance visuelle, dont les limites sont explicites : quatre classes,
images de laboratoire, absence d'explicabilité et de détection d'inconnu. Les perspectives
identifiées — Grad-CAM, calibration, validation croisée, données de terrain, quantification
— tracent un chemin réaliste vers un outil plus robuste et réellement utilisable sur le
terrain.

Sur le plan pédagogique, ce projet a permis de mettre en pratique l'ensemble de la chaîne
d'un projet de Deep Learning : préparation et audit des données, choix d'architecture,
Transfer Learning, régularisation, évaluation critique, intégration applicative, tests et
documentation. Il illustre une idée essentielle : la performance d'un modèle dépend moins
de sa complexité que de la qualité des données, de la rigueur du protocole et de
l'honnêteté de l'interprétation.

## 17. Références bibliographiques

1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant
   health to enable the development of mobile disease diagnostics*. arXiv:1511.08060.
2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for
   image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
   https://doi.org/10.3389/fpls.2016.01419
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2:
   Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on
   Computer Vision and Pattern Recognition (CVPR)*, 4510–4520.
4. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto,
   M., & Adam, H. (2017). *MobileNets: Efficient convolutional neural networks for mobile
   vision applications*. arXiv:1704.04861.
5. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553),
   436–444.
6. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning
   applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
7. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with
   deep convolutional neural networks. *Advances in Neural Information Processing
   Systems (NeurIPS)*, 25, 1097–1105.
8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
   recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern
   Recognition (CVPR)*, 770–778.
9. Simonyan, K., & Zisserman, A. (2015). *Very deep convolutional networks for
   large-scale image recognition*. arXiv:1409.1556.
10. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D.,
    Vanhoucke, V., & Rabinovich, A. (2015). Going deeper with convolutions.
    *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR)*, 1–9.
11. Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on
    Knowledge and Data Engineering*, 22(10), 1345–1359.
12. Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for
    deep learning. *Journal of Big Data*, 6(1), 60.
13. Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization*.
    arXiv:1412.6980.
14. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations
    by back-propagating errors. *Nature*, 323(6088), 533–536.
15. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
    Dropout: A simple way to prevent neural networks from overfitting. *Journal of
    Machine Learning Research*, 15(1), 1929–1958.
16. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).
    Grad-CAM: Visual explanations from deep networks via gradient-based localization.
    *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*,
    618–626.
17. Chollet, F. (2021). *Deep Learning with Python* (2ᵉ édition). Manning Publications.
18. Abadi, M. et al. (2016). TensorFlow: A system for large-scale machine learning.
    *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*,
    265–283.

[[PAGEBREAK]]

## Annexe — Ressources du projet

| Ressource | Emplacement |
| --- | --- |
| Code source complet | dépôt Git du projet |
| Notebook d'expérimentation | `notebooks/plant_disease_training.ipynb` |
| Script de téléchargement du dataset | `training/download_dataset.py` |
| Préparation et découpage | `training/prepare_data.py` |
| Entraînement | `training/train.py` |
| Évaluation et figures | `training/evaluate.py` |
| Prédiction en ligne de commande | `training/predict.py` |
| Modèle entraîné | `models/best_model.keras` |
| Configuration et comptages | `models/training_config.json` |
| Métriques détaillées | `results/metrics/evaluation.json` |
| Rapport de classification | `results/metrics/classification_report.txt` |
| Figures | `results/figures/` |
| Prédictions et erreurs | `results/predictions/` |
| Tests automatisés | `tests/` |
