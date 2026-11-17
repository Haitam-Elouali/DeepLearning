# Fiche expérimentale — Détection des maladies des feuilles (PlantVillage, 38 classes)

**Modèle** : MobileNetV2 + Transfer Learning  
**Expérience réalisée le** : 2026-11-17  
**Environnement** : Python 3.12.10, TensorFlow 2.21.0, CPU (sans GPU)  
**Fiche générée le** : 17/11/2026

## 1. Paramètres d'entraînement

| Paramètre | Valeur |
| --- | --- |
| Taille d'image | 224 × 224 × 3 |
| Batch size | 32 |
| Optimiseur | Adam |
| Perte | categorical_crossentropy |
| Époques phase 1 (effectuées / max) | 30 / 30 |
| Learning rate phase 1 | 0.001 |
| Époques phase 2 (fine-tuning) | 15 |
| Learning rate phase 2 | 1e-05 |
| Couches examinées en fin de base | 30 |
| Couches dégelées (drapeau trainable) | 19 |
| Couches réellement réentraînées (porteuses de poids) | 10 |
| BatchNormalization maintenues gelées | 52 / 52 |
| Graine aléatoire | 42 |
| Early Stopping | patience 5 sur val_loss |
| Model Checkpoint | meilleur val_accuracy (best_model.keras) |
| Paramètres totaux | 2 426 854 |
| Paramètres entraînables (phase 1) | 168 870 |

Data Augmentation : rotation ±5 %, zoom ±10 %, retournement horizontal, luminosité ±15 %, translation ±5 %.

## 2. Effectifs par classe

| Classe | Dataset brut | Entraînement | Validation | Test |
| --- | --- | --- | --- | --- |
| Pommier — Tavelure (Apple — Apple scab) | 630 | 442 | 94 | 94 |
| Pommier — Pourriture noire (Apple — Black rot) | 621 | 435 | 93 | 93 |
| Pommier — Rouille du cèdre (Apple — Cedar apple rust) | 275 | 193 | 41 | 41 |
| Pommier — Feuille saine (Apple — healthy) | 1645 | 1151 | 247 | 247 |
| Myrtillier — Feuille saine (Blueberry — healthy) | 1502 | 1052 | 225 | 225 |
| Cerisier — Oïdium (Cherry (including sour) — Powdery mildew) | 1052 | 736 | 158 | 158 |
| Cerisier — Feuille saine (Cherry (including sour) — healthy) | 854 | 598 | 128 | 128 |
| Maïs — Cercosporiose (tache grise) (Corn (maize) — Cercospora leaf spot Gray leaf spot) | 513 | 359 | 77 | 77 |
| Maïs — Rouille commune (Corn (maize) — Common rust) | 1192 | 834 | 179 | 179 |
| Maïs — Helminthosporiose (Corn (maize) — Northern Leaf Blight) | 985 | 689 | 148 | 148 |
| Maïs — Feuille saine (Corn (maize) — healthy) | 1162 | 814 | 174 | 174 |
| Vigne — Pourriture noire (Grape — Black rot) | 1180 | 826 | 177 | 177 |
| Vigne — Esca (Grape — Esca (Black Measles)) | 1383 | 969 | 207 | 207 |
| Vigne — Brûlure des feuilles (Grape — Leaf blight (Isariopsis Leaf Spot)) | 1076 | 754 | 161 | 161 |
| Vigne — Feuille saine (Grape — healthy) | 423 | 297 | 63 | 63 |
| Oranger — Huanglongbing (Orange — Haunglongbing (Citrus greening)) | 5507 | 3855 | 826 | 826 |
| Pêcher — Tache bactérienne (Peach — Bacterial spot) | 2297 | 1607 | 345 | 345 |
| Pêcher — Feuille saine (Peach — healthy) | 360 | 252 | 54 | 54 |
| Poivron — Tache bactérienne (Pepper, bell — Bacterial spot) | 997 | 697 | 150 | 150 |
| Poivron — Feuille saine (Pepper, bell — healthy) | 1478 | 1034 | 222 | 222 |
| Pomme de terre — Alternariose (Potato — Early blight) | 1000 | 700 | 150 | 150 |
| Pomme de terre — Mildiou (Potato — Late blight) | 1000 | 700 | 150 | 150 |
| Pomme de terre — Feuille saine (Potato — healthy) | 152 | 92 | 30 | 30 |
| Framboisier — Feuille saine (Raspberry — healthy) | 371 | 259 | 56 | 56 |
| Soja — Feuille saine (Soybean — healthy) | 5090 | 3562 | 764 | 764 |
| Courge — Oïdium (Squash — Powdery mildew) | 1835 | 1285 | 275 | 275 |
| Fraisier — Brûlure des feuilles (Strawberry — Leaf scorch) | 1109 | 777 | 166 | 166 |
| Fraisier — Feuille saine (Strawberry — healthy) | 456 | 320 | 68 | 68 |
| Tomate — Tache bactérienne (Tomato — Bacterial spot) | 2127 | 1489 | 319 | 319 |
| Tomate — Alternariose précoce (Tomato — Early blight) | 1000 | 700 | 150 | 150 |
| Tomate — Mildiou (Tomato — Late blight) | 1909 | 1337 | 286 | 286 |
| Tomate — Moisissure des feuilles (Tomato — Leaf Mold) | 952 | 666 | 143 | 143 |
| Tomate — Septoriose (Tomato — Septoria leaf spot) | 1771 | 1239 | 266 | 266 |
| Tomate — Tétranyque à deux points (Tomato — Spider mites Two-spotted spider mite) | 1676 | 1174 | 251 | 251 |
| Tomate — Tache cible (Tomato — Target Spot) | 1404 | 982 | 211 | 211 |
| Tomate — Virus des feuilles jaunes (Tomato — Tomato Yellow Leaf Curl Virus) | 5357 | 3749 | 804 | 804 |
| Tomate — Virus de la mosaïque (Tomato — Tomato mosaic virus) | 373 | 261 | 56 | 56 |
| Tomate — Feuille saine (Tomato — healthy) | 1591 | 1113 | 239 | 239 |
| **Total** | **54305** | **37999** | **8153** | **8153** |

Découpage stratifié, graine 42, 21 groupes de doublons exacts (MD5 global) regroupés, 0 conflit(s) d'annotation, 196 groupes de quasi-doublons signalés, 0 image(s) illisible(s).

## 3. Courbes d'apprentissage

![Courbes accuracy et loss (entraînement / validation)](../../results/training_curves.png)

Meilleure accuracy de validation : **97,02 %** (époque 39, sur 45 époques).

| Phase | Époques | Meilleure val_accuracy | Durée |
| --- | --- | --- | --- |
| Phase 1 — base gelée | 30 | 95,31 % | 474.77 min |
| Phase 2 — fine-tuning | 15 | 97,02 % | 257.6 min |

## 4. Métriques sur l'ensemble de test

| Métrique | Valeur |
| --- | --- |
| Accuracy | **97,31 %** |
| Precision (macro) | 96,72 % |
| Recall (macro) | 96,68 % |
| F1-score (macro) | **96,65 %** |
| F1-score (pondéré) | 97,30 % |
| Images de test | 8153 |
| Images mal classées | 219 (2,69 %) |

## 5. Matrice de confusion

| Réel \ Prédit | Pommier — Tavelure | Pommier — Pourriture noire | Pommier — Rouille du cèdre | Pommier — Feuille saine | Myrtillier — Feuille saine | Cerisier — Oïdium | Cerisier — Feuille saine | Maïs — Cercosporiose (tache grise) | Maïs — Rouille commune | Maïs — Helminthosporiose | Maïs — Feuille saine | Vigne — Pourriture noire | Vigne — Esca | Vigne — Brûlure des feuilles | Vigne — Feuille saine | Oranger — Huanglongbing | Pêcher — Tache bactérienne | Pêcher — Feuille saine | Poivron — Tache bactérienne | Poivron — Feuille saine | Pomme de terre — Alternariose | Pomme de terre — Mildiou | Pomme de terre — Feuille saine | Framboisier — Feuille saine | Soja — Feuille saine | Courge — Oïdium | Fraisier — Brûlure des feuilles | Fraisier — Feuille saine | Tomate — Tache bactérienne | Tomate — Alternariose précoce | Tomate — Mildiou | Tomate — Moisissure des feuilles | Tomate — Septoriose | Tomate — Tétranyque à deux points | Tomate — Tache cible | Tomate — Virus des feuilles jaunes | Tomate — Virus de la mosaïque | Tomate — Feuille saine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pommier — Tavelure | 86 | 0 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pommier — Pourriture noire | 0 | 91 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pommier — Rouille du cèdre | 0 | 0 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pommier — Feuille saine | 0 | 0 | 0 | 247 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Myrtillier — Feuille saine | 0 | 0 | 0 | 0 | 225 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cerisier — Oïdium | 1 | 0 | 0 | 0 | 0 | 156 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Cerisier — Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Maïs — Cercosporiose (tache grise) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 70 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Maïs — Rouille commune | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 179 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Maïs — Helminthosporiose | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 140 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Maïs — Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 174 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Vigne — Pourriture noire | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 171 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Vigne — Esca | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 206 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Vigne — Brûlure des feuilles | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 161 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Vigne — Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 63 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Oranger — Huanglongbing | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 826 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pêcher — Tache bactérienne | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 344 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Pêcher — Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 54 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Poivron — Tache bactérienne | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 145 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Poivron — Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 218 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pomme de terre — Alternariose | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Pomme de terre — Mildiou | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 142 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Pomme de terre — Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 26 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Framboisier — Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Soja — Feuille saine | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 761 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Courge — Oïdium | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Fraisier — Brûlure des feuilles | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 166 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Fraisier — Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Tomate — Tache bactérienne | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 307 | 1 | 1 | 2 | 4 | 1 | 2 | 1 | 0 | 0 |
| Tomate — Alternariose précoce | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 114 | 9 | 4 | 8 | 2 | 7 | 1 | 0 | 0 |
| Tomate — Mildiou | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 267 | 4 | 2 | 2 | 5 | 0 | 0 | 0 |
| Tomate — Moisissure des feuilles | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 134 | 1 | 5 | 2 | 0 | 0 | 0 |
| Tomate — Septoriose | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 1 | 0 | 6 | 248 | 2 | 4 | 0 | 0 | 1 |
| Tomate — Tétranyque à deux points | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 238 | 4 | 0 | 1 | 6 |
| Tomate — Tache cible | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 2 | 0 | 1 | 1 | 14 | 179 | 0 | 0 | 11 |
| Tomate — Virus des feuilles jaunes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 2 | 1 | 5 | 0 | 785 | 4 | 0 |
| Tomate — Virus de la mosaïque | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 0 |
| Tomate — Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 239 |

![Matrice de confusion (effectifs et proportions)](../../results/confusion_matrix.png)

## 6. Résultats par classe

| Classe | Precision | Recall | F1-score | Support | Bien classées | Confiance moyenne |
| --- | --- | --- | --- | --- | --- | --- |
| Pommier — Tavelure | 96,63 % | 91,49 % | 93,99 % | 94 | 86 | 97,17 % |
| Pommier — Pourriture noire | 100,00 % | 97,85 % | 98,91 % | 93 | 91 | 99,70 % |
| Pommier — Rouille du cèdre | 97,62 % | 100,00 % | 98,80 % | 41 | 41 | 99,98 % |
| Pommier — Feuille saine | 97,24 % | 100,00 % | 98,60 % | 247 | 247 | 99,80 % |
| Myrtillier — Feuille saine | 100,00 % | 100,00 % | 100,00 % | 225 | 225 | 99,37 % |
| Cerisier — Oïdium | 100,00 % | 98,73 % | 99,36 % | 158 | 156 | 99,59 % |
| Cerisier — Feuille saine | 96,24 % | 100,00 % | 98,08 % | 128 | 128 | 100,00 % |
| Maïs — Cercosporiose (tache grise) | 90,91 % | 90,91 % | 90,91 % | 77 | 70 | 91,55 % |
| Maïs — Rouille commune | 100,00 % | 100,00 % | 100,00 % | 179 | 179 | 100,00 % |
| Maïs — Helminthosporiose | 95,24 % | 94,59 % | 94,92 % | 148 | 140 | 93,10 % |
| Maïs — Feuille saine | 99,43 % | 100,00 % | 99,71 % | 174 | 174 | 100,00 % |
| Vigne — Pourriture noire | 99,42 % | 96,61 % | 97,99 % | 177 | 171 | 97,04 % |
| Vigne — Esca | 97,17 % | 99,52 % | 98,33 % | 207 | 206 | 98,64 % |
| Vigne — Brûlure des feuilles | 100,00 % | 100,00 % | 100,00 % | 161 | 161 | 99,98 % |
| Vigne — Feuille saine | 100,00 % | 100,00 % | 100,00 % | 63 | 63 | 99,90 % |
| Oranger — Huanglongbing | 99,88 % | 100,00 % | 99,94 % | 826 | 826 | 99,93 % |
| Pêcher — Tache bactérienne | 99,42 % | 99,71 % | 99,57 % | 345 | 344 | 99,32 % |
| Pêcher — Feuille saine | 100,00 % | 100,00 % | 100,00 % | 54 | 54 | 99,44 % |
| Poivron — Tache bactérienne | 98,64 % | 96,67 % | 97,64 % | 150 | 145 | 96,73 % |
| Poivron — Feuille saine | 99,09 % | 98,20 % | 98,64 % | 222 | 218 | 99,07 % |
| Pomme de terre — Alternariose | 100,00 % | 98,67 % | 99,33 % | 150 | 148 | 99,40 % |
| Pomme de terre — Mildiou | 97,93 % | 94,67 % | 96,27 % | 150 | 142 | 96,05 % |
| Pomme de terre — Feuille saine | 89,66 % | 86,67 % | 88,14 % | 30 | 26 | 94,85 % |
| Framboisier — Feuille saine | 100,00 % | 100,00 % | 100,00 % | 56 | 56 | 99,91 % |
| Soja — Feuille saine | 99,61 % | 99,61 % | 99,61 % | 764 | 761 | 99,31 % |
| Courge — Oïdium | 99,64 % | 100,00 % | 99,82 % | 275 | 275 | 99,95 % |
| Fraisier — Brûlure des feuilles | 100,00 % | 100,00 % | 100,00 % | 166 | 166 | 99,74 % |
| Fraisier — Feuille saine | 100,00 % | 100,00 % | 100,00 % | 68 | 68 | 99,79 % |
| Tomate — Tache bactérienne | 94,75 % | 96,24 % | 95,49 % | 319 | 307 | 96,04 % |
| Tomate — Alternariose précoce | 91,94 % | 76,00 % | 83,21 % | 150 | 114 | 82,23 % |
| Tomate — Mildiou | 94,68 % | 93,36 % | 94,01 % | 286 | 267 | 92,96 % |
| Tomate — Moisissure des feuilles | 87,01 % | 93,71 % | 90,24 % | 143 | 134 | 96,35 % |
| Tomate — Septoriose | 92,88 % | 93,23 % | 93,06 % | 266 | 248 | 90,13 % |
| Tomate — Tétranyque à deux points | 87,82 % | 94,82 % | 91,19 % | 251 | 238 | 93,41 % |
| Tomate — Tache cible | 88,18 % | 84,83 % | 86,47 % | 211 | 179 | 85,78 % |
| Tomate — Virus des feuilles jaunes | 99,75 % | 97,64 % | 98,68 % | 804 | 785 | 98,34 % |
| Tomate — Virus de la mosaïque | 91,80 % | 100,00 % | 95,73 % | 56 | 56 | 98,96 % |
| Tomate — Feuille saine | 92,64 % | 100,00 % | 96,18 % | 239 | 239 | 99,22 % |

![Performances par classe](../../results/figures/per_class_performance.png)

## 7. Confiance et analyse des erreurs

| Indicateur | Valeur |
| --- | --- |
| Confiance moyenne (prédictions correctes) | 98,14 % |
| Confiance moyenne (erreurs) | 72,34 % |
| Confiance médiane | 100,00 % |
| Erreurs | 219 / 8153 (2,69 %) |

| Confusion observée | Nombre de cas |
| --- | --- |
| Tomate — Tache cible → Tomate — Tétranyque à deux points | 14 |
| Tomate — Tache cible → Tomate — Feuille saine | 11 |
| Tomate — Alternariose précoce → Tomate — Mildiou | 9 |
| Tomate — Alternariose précoce → Tomate — Septoriose | 8 |
| Tomate — Virus des feuilles jaunes → Tomate — Tache bactérienne | 7 |

![Distribution de la confiance](../../results/figures/confidence_distribution.png)

![Exemples d'erreurs de classification](../../results/predictions/error_grid.png)

## 8. Observations et difficultés rencontrées

- **Convergence rapide de la phase 1** : dès la première époque, la validation atteint 88,94 %, ce qui confirme la qualité des caractéristiques ImageNet.
- **Le fine-tuning apporte un gain réel de +1,70 point de validation** ; son learning rate très faible (1e-5) est indispensable pour ne pas détruire les poids pré-entraînés.
- **Les classes rares restent fragiles** : « Tomate — Alternariose précoce » obtient le F1-score le plus bas (83,21 %), alors que 15 classes dépassent 99 %.
- **Le déséquilibre pèse sur les moyennes** : l'écart entre F1 pondéré et F1 macro est de 0,65 point, trace du traitement inégal des classes rares.
- **Le modèle est sur-confiant** : la confiance médiane atteint 99,999 % ; cette quasi-certitude justifie l'affichage des probabilités plutôt qu'une classe unique.
- **Budget d'entraînement lourd** : 45 époques au total sur 4 threads CPU (≈ 732 minutes).
- **Dataset de laboratoire** : les images ont un fond uniforme ; les performances mesurées ne préjugent pas du comportement sur des photos de terrain.
- **Doublons** : 21 groupes d'images identiques ont été détectés (empreinte MD5 calculée sur l'ensemble du dataset, toutes classes confondues) et regroupés pour éviter toute fuite ; 0 conflit(s) d'annotation et 196 groupes de quasi-doublons (empreinte perceptuelle) sont signalés pour inspection.

## 9. Conclusion de l'expérience

Le modèle MobileNetV2 adapté par Transfer Learning atteint **97,31 % d'accuracy** et un **F1-score macro de 96,65 %** sur 8153 images de test indépendantes, avec un temps d'inférence d'environ 110 ms par image sur CPU. Ces résultats confirment la faisabilité d'un outil local d'aide à la reconnaissance visuelle, dans les limites du périmètre étudié.

## 10. Fichiers de référence

| Ressource | Fichier |
| --- | --- |
| Configuration complète | `models/training_config.json` |
| Historique des époques | `results/metrics/training_history.json` |
| Rapport de classification | `results/metrics/classification_report.txt` |
| Métriques détaillées | `results/metrics.json` |
| Prédictions du test | `results/predictions/predictions_test.csv` |
| Erreurs détaillées | `results/predictions/erreurs.csv` |
