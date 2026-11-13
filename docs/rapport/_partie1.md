# Détection et classification des maladies des plantes à partir d'images de feuilles à l'aide des réseaux de neurones convolutifs (CNN) et du Transfer Learning

**Application au cas de la feuille de tomate — MobileNetV2, TensorFlow/Keras et Flask**

Projet universitaire de fin d'études — Génie informatique

Réalisé par : **[Prénom NOM]**

Encadré par : **[Nom de l'encadrant]**

Établissement : **[Nom de l'université / de l'école]**

Année universitaire : **2025 – 2026**

[[PAGEBREAK]]

## Sommaire

1. Introduction générale
2. Contexte et problématique
3. Objectifs du projet
4. État de l'art : Deep Learning et réseaux de neurones convolutifs
5. Présentation de MobileNetV2
6. Présentation du Transfer Learning
7. Présentation et exploration du dataset
8. Prétraitement et Data Augmentation
9. Architecture du modèle proposé
10. Configuration et processus d'entraînement
11. Développement de l'application Web
12. Résultats expérimentaux
13. Évaluation et interprétation des métriques
14. Discussion des résultats
15. Limites et perspectives d'amélioration
16. Éthique et impacts du système
17. Conclusion générale
18. Références bibliographiques

> **Note sur l'honnêteté scientifique.** Toutes les valeurs numériques de ce rapport
> proviennent d'un entraînement et d'une évaluation réellement exécutés dans le cadre
> de ce projet, sur le dataset PlantVillage téléchargé depuis sa source publique. Les
> figures de résultats sont générées automatiquement par le script
> `training/evaluate.py` à partir des prédictions du modèle sur l'ensemble de test.
> Aucun résultat n'a été simulé, arrondi artificiellement ou recopié d'une publication.

[[PAGEBREAK]]

## 1. Introduction générale

L'agriculture constitue un secteur économique majeur, particulièrement vulnérable aux
maladies des cultures. Parmi les espèces les plus cultivées, la tomate occupe une place
de premier plan : elle est produite sur tous les continents et représente une source de
revenus importante pour de nombreux exploitants. Or, les maladies foliaires de la tomate
— alternariose précoce, mildiou, moisissure des feuilles, entre autres — peuvent réduire
considérablement les rendements et compromettre une récolte entière en quelques jours.

Le diagnostic de ces maladies repose traditionnellement sur l'expertise humaine :
l'agriculteur, le technicien ou l'agronome observe les feuilles, identifie des motifs
visuels (taches, halos, duvets, nécroses) et en déduit une pathologie probable. Cette
démarche se heurte à plusieurs difficultés : elle exige une formation spécifique, elle
est chronophage, elle dépend de la disponibilité d'un expert et elle reste subjective
lorsque les symptômes sont discrets ou se ressemblent. De plus, un diagnostic tardif
réduit fortement l'efficacité des mesures de protection.

Les progrès récents de l'apprentissage profond (*Deep Learning*) et, en particulier, des
réseaux de neurones convolutifs (*Convolutional Neural Networks*, CNN) offrent une
alternative prometteuse. Ces modèles sont capables d'apprendre automatiquement des
représentations visuelles discriminantes à partir d'exemples étiquetés, sans que l'on
ait à définir manuellement les caractéristiques pertinentes. Appliqués à des
photographies de feuilles, ils peuvent distinguer une feuille saine d'une feuille
malade, et identifier la maladie en cause avec une précision souvent élevée sur des jeux
de données de référence.

Le présent projet s'inscrit dans cette dynamique. Il propose un système complet,
**de la donnée brute à l'application Web**, permettant à un utilisateur de téléverser
une photographie de feuille de tomate et d'obtenir une prédiction parmi quatre classes,
accompagnée d'un score de confiance et de la distribution des probabilités. La démarche
s'appuie sur un modèle **MobileNetV2** pré-entraîné sur ImageNet, adapté par
**Transfer Learning** à notre problème à quatre classes, puis intégré dans une
application **Flask** dotée d'une interface en français, d'un historique des analyses et
de fiches descriptives des maladies.

Le rapport est organisé de la manière suivante. Les sections 2 et 3 exposent le contexte,
la problématique et les objectifs. Les sections 4 à 6 présentent le cadre théorique :
CNN, architecture MobileNetV2 et Transfer Learning. La section 7 décrit le dataset et
son exploration, la section 8 le prétraitement et la Data Augmentation. Les sections 9
et 10 détaillent l'architecture du modèle et le protocole d'entraînement. La section 11
est consacrée à l'application Web. Les sections 12 à 14 présentent et discutent les
résultats expérimentaux. La section 15 dresse les limites et les perspectives, la section
16 traite de l'éthique et des impacts du système, la section 17 présente la conclusion.
La section 18 rassemble les références.

## 2. Contexte et problématique

### 2.1 Contexte agricole et scientifique

La tomate (*Solanum lycopersicum*) est sensible à de nombreuses maladies foliaires
d'origine fongique, oomycète, bactérienne ou virale. Trois d'entre elles dominent
largement les préoccupations des producteurs :

- l'**alternariose précoce**, causée par *Alternaria solani*, qui se manifeste par des
  taches brunes concentriques sur les feuilles âgées ;
- le **mildiou**, causé par *Phytophthora infestans*, dont la progression rapide en
  conditions humides peut détruire une parcelle en quelques jours ;
- la **moisissure des feuilles**, causée par *Passalora fulva*, fréquente sous serre et
  caractérisée par un duvet olive sur la face inférieure des feuilles.

À ces trois pathologies s'ajoute la classe « feuille saine », indispensable pour que le
système apprenne à ne pas signaler de maladie en l'absence de symptôme.

Sur le plan scientifique, la reconnaissance automatique de maladies des plantes à partir
d'images est un domaine actif. Le jeu de données **PlantVillage**, publié par Hughes et
Salathé (2015) puis exploité par Mohanty, Hughes et Salathé (2016), a servi de référence
à de nombreux travaux. Ces études ont montré que des CNN entraînés sur des images de
feuilles peuvent atteindre des précisions supérieures à 99 % sur ce jeu de données, tout
en soulignant une limite essentielle : les performances chutent lorsque les images
proviennent de conditions réelles, différentes de celles du laboratoire.

### 2.2 Problématique

La problématique du projet peut être formulée ainsi :

> **Peut-on concevoir un système accessible, fonctionnant localement et sans service
> payant, capable de classer automatiquement une photographie de feuille de tomate parmi
> quatre catégories (saine, alternariose précoce, mildiou, moisissure des feuilles), avec
> une précision utile, un temps de réponse de l'ordre de la seconde et une interface
> compréhensible par un utilisateur non spécialiste ?**

Cette problématique soulève plusieurs sous-questions techniques :

1. Comment préparer un jeu de données fiable, équilibré et sans fuite entre entraînement
   et évaluation ?
2. Quel modèle offre le meilleur compromis entre précision et coût de calcul, sachant
   que l'application doit fonctionner sur un ordinateur sans GPU ?
3. Comment garantir que le prétraitement appliqué lors de la prédiction soit identique à
   celui utilisé pendant l'entraînement ?
4. Comment présenter le résultat de manière honnête, en affichant la distribution des
   probabilités plutôt qu'une classe unique présentée comme une certitude ?
5. Comment intégrer le modèle dans une application Web robuste, sécurisée et maintenable ?

### 2.3 Enjeu et portée

L'enjeu n'est pas de remplacer l'expertise agronomique, mais de fournir un **outil d'aide
à la reconnaissance visuelle**, utilisable sur le terrain comme en formation. Le système
doit donc être rapide, compréhensible et explicitement présenté comme expérimental. Cette
posture est essentielle : un modèle entraîné sur des images de laboratoire ne peut pas
prétendre à un diagnostic fiable sur des photographies de terrain.

## 3. Objectifs du projet

### 3.1 Objectif général

Concevoir, entraîner, évaluer et déployer localement une application Web de détection et
de classification des maladies de la feuille de tomate, fondée sur un CNN et le Transfer
Learning.

### 3.2 Objectifs spécifiques

**Sur le plan des données**

- télécharger et organiser le sous-ensemble tomate du dataset PlantVillage ;
- vérifier l'intégrité des images et détecter les doublons ;
- construire un découpage stratifié entraînement / validation / test (70 / 15 / 15)
  reproductible et sans fuite de données ;
- documenter précisément la provenance, les effectifs et les limites du dataset.

**Sur le plan de la modélisation**

- construire un modèle MobileNetV2 pré-entraîné sur ImageNet, adapté à quatre classes ;
- entraîner la tête de classification en base gelée, puis réaliser un fine-tuning partiel ;
- mettre en place les mécanismes de régularisation et de suivi (Early Stopping, Model
  Checkpoint, réduction du learning rate, journalisation) ;
- sauvegarder le meilleur modèle et l'ensemble de ses hyperparamètres.

**Sur le plan de l'évaluation**

- calculer accuracy, precision, recall et F1-score sur un ensemble de test indépendant ;
- produire la matrice de confusion, le rapport de classification et les courbes
  d'apprentissage ;
- analyser les erreurs et les confusions entre classes ;
- discuter les limites du modèle.

**Sur le plan applicatif**

- développer une application Flask en français, responsive et moderne ;
- proposer cinq pages : tableau de bord, analyse, historique, fiches maladies, à propos ;
- stocker les analyses dans une base SQLite et permettre leur consultation et leur
  suppression ;
- garantir la sécurité des téléversements et éviter de recharger le modèle à chaque
  requête ;
- fournir des tests automatisés et une documentation d'installation.

### 3.3 Contraintes

Le projet doit rester réalisable par un étudiant en génie informatique, avec des moyens
limités : pas de GPU dédié, pas de service cloud payant, pas d'API externe. Toutes les
étapes doivent pouvoir être rejouées localement à partir du code fourni.

## 4. État de l'art : Deep Learning et réseaux de neurones convolutifs

### 4.1 Du perceptron aux réseaux profonds

Un réseau de neurones artificiel est composé de couches de neurones interconnectés.
Chaque neurone calcule une combinaison linéaire de ses entrées, à laquelle il applique
une fonction d'activation non linéaire. L'empilement de plusieurs couches permet
d'apprendre des fonctions de plus en plus abstraites. L'apprentissage consiste à ajuster
les poids du réseau afin de minimiser une fonction de perte mesurant l'écart entre les
prédictions et les étiquettes réelles, généralement par **descente de gradient** et
**rétropropagation du gradient** (Rumelhart, Hinton et Williams, 1986).

Les réseaux entièrement connectés appliqués directement à des images se heurteraient à
un problème de dimension : une image de 224 × 224 × 3 contient plus de 150 000 valeurs,
et une première couche de 1 000 neurones nécessiterait plus de 150 millions de poids.
Les réseaux convolutifs résolvent ce problème en exploitant la structure spatiale des
images.

### 4.2 Principes des réseaux convolutifs

**La convolution.** Une couche de convolution applique un ensemble de filtres (noyaux) de
petite taille sur l'image d'entrée. Chaque filtre produit une carte d'activation qui
répond à un motif local : contour, texture, couleur. Un filtre de 3 × 3 partagé sur toute
l'image ne possède que 27 poids, ce qui réduit considérablement le nombre de paramètres
tout en introduisant deux propriétés fondamentales :

- le **partage des poids** : le même détecteur est appliqué partout dans l'image ;
- l'**invariance par translation** : un motif reconnu reste reconnu s'il se déplace.

**Le pooling.** Les couches de *max pooling* réduisent la résolution spatiale en
conservant la réponse la plus forte de chaque région. Elles diminuent le coût de calcul
et rendent la représentation plus robuste aux petites translations.

**Les fonctions d'activation.** La fonction ReLU (`max(0, x)`) est la plus utilisée : elle
introduit la non-linéarité nécessaire tout en limitant le problème de disparition du
gradient. La couche de sortie utilise une fonction **softmax**, qui transforme un vecteur
de scores en une distribution de probabilités dont la somme vaut 1 :

p_i = exp(z_i) / somme_j exp(z_j), pour i = 1 à 4.

**La perte et l'optimisation.** Pour une classification multi-classes, la perte usuelle
est l'**entropie croisée catégorielle**. L'optimiseur Adam (Kingma et Ba, 2015) adapte le
pas d'apprentissage de chaque paramètre à partir des moments du gradient, ce qui le rend
robuste et rapide à converger, avec peu de réglage manuel.

### 4.3 Architectures marquantes

| Architecture | Année | Apport principal |
| --- | --- | --- |
| LeNet-5 | 1998 | Premiers succès de la convolution sur les chiffres manuscrits |
| AlexNet | 2012 | Victoire d'ImageNet ; popularisation de ReLU, dropout et GPU |
| VGG-16/19 | 2014 | Empilement régulier de convolutions 3 × 3 ; simplicité conceptuelle |
| ResNet | 2015 | Connexions résiduelles permettant d'entraîner des réseaux très profonds |
| Inception | 2014 | Convolutions multi-échelles parallèles |
| MobileNet (V1/V2) | 2017-2018 | Convolutions séparables en profondeur, pour les appareils à faibles ressources |

*Tableau 1 — Architectures convolutives marquantes.*

Notre choix s'est porté sur MobileNetV2, pour des raisons de coût de calcul et de
pertinence pratique : l'application doit fonctionner sur un ordinateur portable sans GPU
et produire une prédiction en moins d'une seconde.

### 4.4 Régularisation et prévention du sur-apprentissage

Le sur-apprentissage (*overfitting*) désigne la situation où le modèle mémorise les
exemples d'entraînement sans généraliser. Plusieurs techniques sont utilisées dans ce
projet :

- **Data Augmentation** : création de variantes artificielles des images d'entraînement ;
- **Dropout** : désactivation aléatoire de neurones pendant l'entraînement ;
- **Early Stopping** : arrêt de l'entraînement lorsque la perte de validation cesse de
  s'améliorer ;
- **réduction du learning rate** : diminution du pas lorsque la validation stagne ;
- **poids de classes** : compensation du déséquilibre entre classes ;
- **gel des couches pré-entraînées** : limite le nombre de paramètres à apprendre.

### 4.5 Métriques d'évaluation

Soit TP, FP, FN et TN les nombres de vrais positifs, faux positifs, faux négatifs et vrais
négatifs pour une classe donnée :

- **Accuracy** : proportion de prédictions correctes, toutes classes confondues ;
- **Precision** : TP / (TP + FP), capacité à ne pas signaler une classe à tort ;
- **Recall** : TP / (TP + FN), capacité à retrouver tous les cas de la classe ;
- **F1-score** : moyenne harmonique de la precision et du recall.

L'accuracy seule est insuffisante en cas de déséquilibre : un modèle qui prédirait
toujours la classe majoritaire obtiendrait une accuracy trompeuse. C'est pourquoi la
matrice de confusion et les métriques par classe sont systématiquement analysées.

## 5. Présentation de MobileNetV2

### 5.1 Motivation : l'efficacité computationnelle

MobileNetV2, proposé par Sandler et al. (2018), est une évolution de MobileNet (Howard
et al., 2017). Son objectif est de fournir un réseau convolutif performant avec un coût
de calcul et un nombre de paramètres très inférieurs à ceux de VGG ou ResNet, afin de
permettre une exécution sur téléphone ou micro-ordinateur. Notre modèle complet ne
compte ainsi que **2 422 468 paramètres**, dont seulement **164 484** sont entraînés lors
de la première phase.

### 5.2 Convolutions séparables en profondeur

Une convolution standard de `Dk × Dk` noyaux appliquée à `M` canaux d'entrée pour
produire `N` canaux coûte `Dk² × M × N` multiplications par position. MobileNet la
remplace par deux opérations :

1. une **convolution en profondeur** (*depthwise*) : un filtre par canal d'entrée, coût
   `Dk² × M` ;
2. une **convolution ponctuelle** (*pointwise*) 1 × 1 qui recombine les canaux, coût
   `M × N`.

Le rapport de coût est approximativement `1/N + 1/Dk²`, soit une réduction d'environ 8 à
9 fois pour des noyaux 3 × 3. C'est cette économie qui rend le modèle compatible avec une
exécution sur processeur.

### 5.3 Blocs résiduels inversés et goulot linéaire

MobileNetV2 introduit deux idées structurelles :

- le **bloc résiduel inversé** (*inverted residual*) : contrairement à ResNet, la
  connexion résiduelle relie des représentations de faible dimension, tandis que la
  couche intermédiaire est élargie (facteur d'expansion 6) ;
- le **goulot linéaire** (*linear bottleneck*) : la dernière convolution 1 × 1 de chaque
  bloc n'applique pas de ReLU, car une activation non linéaire détruirait l'information
  dans un espace de faible dimension.

Le réseau alterne ainsi des blocs d'expansion, de convolution en profondeur et de
projection, pour un total de 53 couches convolutives dans la version 1.0 de largeur 224.

### 5.4 Sortie et adaptation à notre problème

La version utilisée est `MobileNetV2(weights="imagenet", include_top=False)`, avec une
entrée de 224 × 224 × 3. La couche de classification d'origine (1 000 classes ImageNet)
est supprimée. La sortie de la base est un tenseur de forme 7 × 7 × 1 280, que nous
agrégeons par un *Global Average Pooling* pour obtenir un vecteur de 1 280
caractéristiques. Ce vecteur alimente ensuite notre propre tête de classification.

### 5.5 Pourquoi ce choix pour notre projet

- **coût maîtrisé** : entraînement et inférence possibles sur CPU ;
- **taille raisonnable** : un fichier modèle de quelques dizaines de mégaoctets ;
- **qualité des caractéristiques ImageNet** : textures, contours et couleurs déjà
  apprises, très utiles pour analyser des feuilles ;
- **latence faible** : environ 70 ms par image mesurés sur notre environnement de test.

## 6. Présentation du Transfer Learning

### 6.1 Principe

Le Transfer Learning consiste à réutiliser un modèle entraîné sur une tâche source
(généralement ImageNet, 1,2 million d'images et 1 000 classes) pour résoudre une tâche
cible disposant de moins de données. Les premières couches d'un CNN apprennent des
motifs génériques (contours, couleurs, textures) qui restent pertinents pour de
nombreuses tâches visuelles. Seules les couches les plus profondes, spécialisées dans les
classes d'origine, doivent être remplacées.

Cette approche est particulièrement adaptée ici : notre dataset ne contient que
5 452 images, un volume insuffisant pour entraîner un CNN profond from scratch sans
sur-apprentissage sévère.

### 6.2 Deux stratégies complémentaires

**Extraction de caractéristiques (feature extraction).** La base pré-entraînée est
gelée (`trainable = False`). Seule la nouvelle tête de classification est entraînée. Le
nombre de paramètres apprenables est faible (164 484 dans notre cas), l'entraînement est
rapide et le risque de sur-apprentissage limité.

**Fine-Tuning.** Les dernières couches de la base sont dégelées et ré-entraînées avec un
learning rate très faible (1e-5 dans notre cas), afin d'adapter les filtres les plus
spécifiques aux textures et couleurs de nos feuilles. Un learning rate élevé détruirait
les poids pré-entraînés : c'est pourquoi il doit rester au moins dix fois plus petit que
celui de la phase 1.

Dans notre projet, les couches **BatchNormalization** restent gelées pendant le
fine-tuning. Leur mise à jour avec de petits lots (batch de 32) déstabilise
l'apprentissage, car les statistiques de normalisation sont estimées sur trop peu
d'exemples.

### 6.3 Avantages et limites

| Avantages | Limites |
| --- | --- |
| Apprentissage rapide avec peu de données | Dépend de la similarité entre domaine source et cible |
| Meilleure généralisation qu'un modèle from scratch | Risque de biais hérité d'ImageNet |
| Coût de calcul réduit | Fine-tuning délicat (learning rate, couches à dégeler) |
| Base solide pour un projet étudiant | Ne compense pas un dataset trop pauvre ou mal étiqueté |

*Tableau 2 — Avantages et limites du Transfer Learning.*

Le Transfer Learning est donc un levier méthodologique, pas une garantie de performance :
la qualité des données et la rigueur de l'évaluation restent déterminantes.
