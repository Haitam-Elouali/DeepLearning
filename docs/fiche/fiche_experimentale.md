# Fiche expérimentale — Détection des maladies des feuilles (PlantVillage, 38 classes)

**Modèle** : MobileNetV2 + Transfer Learning  
**Expérience réalisée le** : 2026-11-15  
**Environnement** : Python 3.12.10, TensorFlow 2.21.0, CPU (sans GPU)  
**Fiche générée le** : 15/11/2026

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
| Apple — Apple scab (Apple___Apple_scab) | 630 | 442 | 94 | 94 |
| Apple — Black rot (Apple___Black_rot) | 621 | 435 | 93 | 93 |
| Apple — Cedar apple rust (Apple___Cedar_apple_rust) | 275 | 193 | 41 | 41 |
| Apple — healthy (Apple___healthy) | 1645 | 1149 | 248 | 248 |
| Blueberry — healthy (Blueberry___healthy) | 1502 | 1052 | 225 | 225 |
| Cherry (including sour) — Powdery mildew (Cherry_(including_sour)___Powdery_mildew) | 1052 | 736 | 158 | 158 |
| Cherry (including sour) — healthy (Cherry_(including_sour)___healthy) | 854 | 598 | 128 | 128 |
| Corn (maize) — Cercospora leaf spot Gray leaf spot (Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot) | 513 | 359 | 77 | 77 |
| Corn (maize) — Common rust (Corn_(maize)___Common_rust_) | 1192 | 834 | 179 | 179 |
| Corn (maize) — Northern Leaf Blight (Corn_(maize)___Northern_Leaf_Blight) | 985 | 689 | 148 | 148 |
| Corn (maize) — healthy (Corn_(maize)___healthy) | 1162 | 814 | 174 | 174 |
| Grape — Black rot (Grape___Black_rot) | 1180 | 826 | 177 | 177 |
| Grape — Esca (Black Measles) (Grape___Esca_(Black_Measles)) | 1383 | 969 | 207 | 207 |
| Grape — Leaf blight (Isariopsis Leaf Spot) (Grape___Leaf_blight_(Isariopsis_Leaf_Spot)) | 1076 | 754 | 161 | 161 |
| Grape — healthy (Grape___healthy) | 423 | 297 | 63 | 63 |
| Orange — Haunglongbing (Citrus greening) (Orange___Haunglongbing_(Citrus_greening)) | 5507 | 3855 | 826 | 826 |
| Peach — Bacterial spot (Peach___Bacterial_spot) | 2297 | 1607 | 345 | 345 |
| Peach — healthy (Peach___healthy) | 360 | 252 | 54 | 54 |
| Pepper, bell — Bacterial spot (Pepper,_bell___Bacterial_spot) | 997 | 697 | 150 | 150 |
| Pepper, bell — healthy (Pepper,_bell___healthy) | 1478 | 1034 | 222 | 222 |
| Potato — Early blight (Potato___Early_blight) | 1000 | 700 | 150 | 150 |
| Potato — Late blight (Potato___Late_blight) | 1000 | 700 | 150 | 150 |
| Potato — healthy (Potato___healthy) | 152 | 92 | 30 | 30 |
| Raspberry — healthy (Raspberry___healthy) | 371 | 259 | 56 | 56 |
| Soybean — healthy (Soybean___healthy) | 5090 | 3562 | 764 | 764 |
| Squash — Powdery mildew (Squash___Powdery_mildew) | 1835 | 1285 | 275 | 275 |
| Strawberry — Leaf scorch (Strawberry___Leaf_scorch) | 1109 | 777 | 166 | 166 |
| Strawberry — healthy (Strawberry___healthy) | 456 | 320 | 68 | 68 |
| Tomato — Bacterial spot (Tomato___Bacterial_spot) | 2127 | 1489 | 319 | 319 |
| Alternariose précoce (Tomato___Early_blight) | 1000 | 700 | 150 | 150 |
| Mildiou (Tomato___Late_blight) | 1909 | 1334 | 288 | 287 |
| Moisissure des feuilles (Tomato___Leaf_Mold) | 952 | 666 | 143 | 143 |
| Tomato — Septoria leaf spot (Tomato___Septoria_leaf_spot) | 1771 | 1239 | 266 | 266 |
| Tomato — Spider mites Two-spotted spider mite (Tomato___Spider_mites Two-spotted_spider_mite) | 1676 | 1174 | 251 | 251 |
| Tomato — Target Spot (Tomato___Target_Spot) | 1404 | 982 | 211 | 211 |
| Tomato — Tomato Yellow Leaf Curl Virus (Tomato___Tomato_Yellow_Leaf_Curl_Virus) | 5357 | 3749 | 804 | 804 |
| Tomato — Tomato mosaic virus (Tomato___Tomato_mosaic_virus) | 373 | 261 | 56 | 56 |
| Feuille saine (Tomato___healthy) | 1591 | 1111 | 240 | 240 |
| **Total** | **54305** | **37992** | **8157** | **8156** |

Découpage stratifié, graine 42, 21 groupes de doublons exacts (MD5 global) regroupés, 0 conflit(s) d'annotation, 196 groupes de quasi-doublons signalés, 0 image(s) illisible(s).

## 3. Courbes d'apprentissage

![Courbes accuracy et loss (entraînement / validation)](../../results/figures/training_curves.png)

Meilleure accuracy de validation : **97,01 %** (époque 45, sur 45 époques).

| Phase | Époques | Meilleure val_accuracy | Durée |
| --- | --- | --- | --- |
| Phase 1 — base gelée | 30 | 94,75 % | 498.13 min |
| Phase 2 — fine-tuning | 15 | 97,01 % | 267.25 min |

## 4. Métriques sur l'ensemble de test

| Métrique | Valeur |
| --- | --- |
| Accuracy | **97,06 %** |
| Precision (macro) | 95,98 % |
| Recall (macro) | 96,66 % |
| F1-score (macro) | **96,17 %** |
| F1-score (pondéré) | 97,05 % |
| Images de test | 8156 |
| Images mal classées | 240 (2,94 %) |

## 5. Matrice de confusion

| Réel \ Prédit | Apple — Apple scab | Apple — Black rot | Apple — Cedar apple rust | Apple — healthy | Blueberry — healthy | Cherry (including sour) — Powdery mildew | Cherry (including sour) — healthy | Corn (maize) — Cercospora leaf spot Gray leaf spot | Corn (maize) — Common rust | Corn (maize) — Northern Leaf Blight | Corn (maize) — healthy | Grape — Black rot | Grape — Esca (Black Measles) | Grape — Leaf blight (Isariopsis Leaf Spot) | Grape — healthy | Orange — Haunglongbing (Citrus greening) | Peach — Bacterial spot | Peach — healthy | Pepper, bell — Bacterial spot | Pepper, bell — healthy | Potato — Early blight | Potato — Late blight | Potato — healthy | Raspberry — healthy | Soybean — healthy | Squash — Powdery mildew | Strawberry — Leaf scorch | Strawberry — healthy | Tomato — Bacterial spot | Alternariose précoce | Mildiou | Moisissure des feuilles | Tomato — Septoria leaf spot | Tomato — Spider mites Two-spotted spider mite | Tomato — Target Spot | Tomato — Tomato Yellow Leaf Curl Virus | Tomato — Tomato mosaic virus | Feuille saine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Apple — Apple scab | 92 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple — Black rot | 0 | 93 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple — Cedar apple rust | 0 | 0 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple — healthy | 1 | 2 | 0 | 242 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Blueberry — healthy | 0 | 0 | 0 | 1 | 224 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cherry (including sour) — Powdery mildew | 0 | 0 | 0 | 0 | 0 | 158 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cherry (including sour) — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 127 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn (maize) — Cercospora leaf spot Gray leaf spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 73 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn (maize) — Common rust | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 179 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn (maize) — Northern Leaf Blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 117 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn (maize) — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 174 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape — Black rot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 175 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape — Esca (Black Measles) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 205 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape — Leaf blight (Isariopsis Leaf Spot) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 160 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 63 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Orange — Haunglongbing (Citrus greening) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 826 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Peach — Bacterial spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 343 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Peach — healthy | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pepper, bell — Bacterial spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 145 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pepper, bell — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 218 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Potato — Early blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 144 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Potato — Late blight | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 138 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Potato — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 29 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Raspberry — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Soybean — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 762 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Squash — Powdery mildew | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strawberry — Leaf scorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 166 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strawberry — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Tomato — Bacterial spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 305 | 2 | 2 | 0 | 8 | 0 | 0 | 0 | 0 | 1 |
| Alternariose précoce | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 106 | 7 | 4 | 6 | 2 | 14 | 0 | 1 | 1 |
| Mildiou | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 278 | 3 | 0 | 2 | 0 | 0 | 0 | 0 |
| Moisissure des feuilles | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 136 | 3 | 2 | 1 | 0 | 1 | 0 |
| Tomato — Septoria leaf spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 2 | 0 | 3 | 249 | 0 | 4 | 0 | 0 | 2 |
| Tomato — Spider mites Two-spotted spider mite | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 239 | 8 | 0 | 1 | 2 |
| Tomato — Target Spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 9 | 189 | 0 | 0 | 8 |
| Tomato — Tomato Yellow Leaf Curl Virus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 1 | 5 | 1 | 7 | 4 | 774 | 1 | 0 |
| Tomato — Tomato mosaic virus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 55 | 0 |
| Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 239 |

![Matrice de confusion (effectifs et proportions)](../../results/figures/confusion_matrix.png)

## 6. Résultats par classe

| Classe | Precision | Recall | F1-score | Support | Bien classées | Confiance moyenne |
| --- | --- | --- | --- | --- | --- | --- |
| Apple — Apple scab | 96,84 % | 97,87 % | 97,35 % | 94 | 92 | 98,71 % |
| Apple — Black rot | 97,89 % | 100,00 % | 98,94 % | 93 | 93 | 99,99 % |
| Apple — Cedar apple rust | 100,00 % | 100,00 % | 100,00 % | 41 | 41 | 99,95 % |
| Apple — healthy | 98,78 % | 97,58 % | 98,17 % | 248 | 242 | 99,13 % |
| Blueberry — healthy | 100,00 % | 99,56 % | 99,78 % | 225 | 224 | 98,74 % |
| Cherry (including sour) — Powdery mildew | 100,00 % | 100,00 % | 100,00 % | 158 | 158 | 99,76 % |
| Cherry (including sour) — healthy | 97,69 % | 99,22 % | 98,45 % | 128 | 127 | 99,79 % |
| Corn (maize) — Cercospora leaf spot Gray leaf spot | 70,87 % | 94,81 % | 81,11 % | 77 | 73 | 95,22 % |
| Corn (maize) — Common rust | 98,90 % | 100,00 % | 99,44 % | 179 | 179 | 100,00 % |
| Corn (maize) — Northern Leaf Blight | 98,32 % | 79,05 % | 87,64 % | 148 | 117 | 90,11 % |
| Corn (maize) — healthy | 99,43 % | 100,00 % | 99,71 % | 174 | 174 | 100,00 % |
| Grape — Black rot | 98,87 % | 98,87 % | 98,87 % | 177 | 175 | 98,84 % |
| Grape — Esca (Black Measles) | 98,56 % | 99,03 % | 98,80 % | 207 | 205 | 99,03 % |
| Grape — Leaf blight (Isariopsis Leaf Spot) | 98,77 % | 99,38 % | 99,07 % | 161 | 160 | 99,78 % |
| Grape — healthy | 98,44 % | 100,00 % | 99,21 % | 63 | 63 | 100,00 % |
| Orange — Haunglongbing (Citrus greening) | 100,00 % | 100,00 % | 100,00 % | 826 | 826 | 99,97 % |
| Peach — Bacterial spot | 99,71 % | 99,42 % | 99,56 % | 345 | 343 | 99,58 % |
| Peach — healthy | 96,36 % | 98,15 % | 97,25 % | 54 | 53 | 98,83 % |
| Pepper, bell — Bacterial spot | 97,32 % | 96,67 % | 96,99 % | 150 | 145 | 97,87 % |
| Pepper, bell — healthy | 97,76 % | 98,20 % | 97,98 % | 222 | 218 | 98,03 % |
| Potato — Early blight | 97,96 % | 96,00 % | 96,97 % | 150 | 144 | 97,86 % |
| Potato — Late blight | 95,17 % | 92,00 % | 93,56 % | 150 | 138 | 95,30 % |
| Potato — healthy | 82,86 % | 96,67 % | 89,23 % | 30 | 29 | 98,15 % |
| Raspberry — healthy | 100,00 % | 100,00 % | 100,00 % | 56 | 56 | 99,76 % |
| Soybean — healthy | 99,74 % | 99,74 % | 99,74 % | 764 | 762 | 99,67 % |
| Squash — Powdery mildew | 100,00 % | 100,00 % | 100,00 % | 275 | 275 | 99,98 % |
| Strawberry — Leaf scorch | 99,40 % | 100,00 % | 99,70 % | 166 | 166 | 99,66 % |
| Strawberry — healthy | 100,00 % | 100,00 % | 100,00 % | 68 | 68 | 99,99 % |
| Tomato — Bacterial spot | 93,56 % | 95,61 % | 94,57 % | 319 | 305 | 96,10 % |
| Alternariose précoce | 93,81 % | 70,67 % | 80,61 % | 150 | 106 | 84,56 % |
| Mildiou | 94,88 % | 96,86 % | 95,86 % | 287 | 278 | 96,39 % |
| Moisissure des feuilles | 88,89 % | 95,10 % | 91,89 % | 143 | 136 | 95,70 % |
| Tomato — Septoria leaf spot | 92,57 % | 93,61 % | 93,08 % | 266 | 249 | 92,63 % |
| Tomato — Spider mites Two-spotted spider mite | 90,53 % | 95,22 % | 92,82 % | 251 | 239 | 93,88 % |
| Tomato — Target Spot | 85,91 % | 89,57 % | 87,70 % | 211 | 189 | 91,24 % |
| Tomato — Tomato Yellow Leaf Curl Virus | 100,00 % | 96,27 % | 98,10 % | 804 | 774 | 98,25 % |
| Tomato — Tomato mosaic virus | 93,22 % | 98,21 % | 95,65 % | 56 | 55 | 97,09 % |
| Feuille saine | 94,09 % | 99,58 % | 96,76 % | 240 | 239 | 98,69 % |

![Performances par classe](../../results/figures/per_class_performance.png)

## 7. Confiance et analyse des erreurs

| Indicateur | Valeur |
| --- | --- |
| Confiance moyenne (prédictions correctes) | 98,53 % |
| Confiance moyenne (erreurs) | 74,44 % |
| Confiance médiane | 100,00 % |
| Erreurs | 240 / 8156 (2,94 %) |

| Confusion observée | Nombre de cas |
| --- | --- |
| Corn (maize) — Northern Leaf Blight → Corn (maize) — Cercospora leaf spot Gray leaf spot | 30 |
| Alternariose précoce → Tomato — Target Spot | 14 |
| Tomato — Tomato Yellow Leaf Curl Virus → Tomato — Bacterial spot | 11 |
| Tomato — Target Spot → Tomato — Spider mites Two-spotted spider mite | 9 |
| Tomato — Target Spot → Feuille saine | 8 |

![Distribution de la confiance](../../results/figures/confidence_distribution.png)

![Exemples d'erreurs de classification](../../results/predictions/error_grid.png)

## 8. Observations et difficultés rencontrées

- **Convergence rapide de la phase 1** : dès la première époque, la validation atteint 86,59 %, ce qui confirme la qualité des caractéristiques ImageNet.
- **Le fine-tuning apporte un gain réel de +2,26 point de validation** ; son learning rate très faible (1e-5) est indispensable pour ne pas détruire les poids pré-entraînés.
- **Les classes rares restent fragiles** : « Alternariose précoce » obtient le F1-score le plus bas (80,61 %), alors que 14 classes dépassent 99 %.
- **Le déséquilibre pèse sur les moyennes** : l'écart entre F1 pondéré et F1 macro est de 0,87 point, trace du traitement inégal des classes rares.
- **Le modèle est sur-confiant** : la confiance médiane atteint 99,999 % ; cette quasi-certitude justifie l'affichage des probabilités plutôt qu'une classe unique.
- **Budget d'entraînement lourd** : 45 époques au total sur 4 threads CPU (≈ 765 minutes).
- **Dataset de laboratoire** : les images ont un fond uniforme ; les performances mesurées ne préjugent pas du comportement sur des photos de terrain.
- **Doublons** : 21 groupes d'images identiques ont été détectés (empreinte MD5 calculée sur l'ensemble du dataset, toutes classes confondues) et regroupés pour éviter toute fuite ; 0 conflit(s) d'annotation et 196 groupes de quasi-doublons (empreinte perceptuelle) sont signalés pour inspection.

## 9. Conclusion de l'expérience

Le modèle MobileNetV2 adapté par Transfer Learning atteint **97,06 % d'accuracy** et un **F1-score macro de 96,17 %** sur 8156 images de test indépendantes, avec un temps d'inférence d'environ 110 ms par image sur CPU. Ces résultats confirment la faisabilité d'un outil local d'aide à la reconnaissance visuelle, dans les limites du périmètre étudié.

## 10. Fichiers de référence

| Ressource | Fichier |
| --- | --- |
| Configuration complète | `models/training_config.json` |
| Historique des époques | `results/metrics/training_history.json` |
| Rapport de classification | `results/metrics/classification_report.txt` |
| Métriques détaillées | `results/metrics/metrics.json` |
| Prédictions du test | `results/predictions/predictions_test.csv` |
| Erreurs détaillées | `results/predictions/erreurs.csv` |
