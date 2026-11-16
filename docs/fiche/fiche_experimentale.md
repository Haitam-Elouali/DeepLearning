# Fiche expérimentale — Détection des maladies des feuilles (PlantVillage, 38 classes)

**Modèle** : MobileNetV2 + Transfer Learning  
**Expérience réalisée le** : 2026-11-15  
**Environnement** : Python 3.12.10, TensorFlow 2.21.0, CPU (sans GPU)  
**Fiche générée le** : 16/11/2026

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
| Apple — healthy (Apple___healthy) | 1645 | 1151 | 247 | 247 |
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
| Mildiou (Tomato___Late_blight) | 1909 | 1337 | 286 | 286 |
| Moisissure des feuilles (Tomato___Leaf_Mold) | 952 | 666 | 143 | 143 |
| Tomato — Septoria leaf spot (Tomato___Septoria_leaf_spot) | 1771 | 1239 | 266 | 266 |
| Tomato — Spider mites Two-spotted spider mite (Tomato___Spider_mites Two-spotted_spider_mite) | 1676 | 1174 | 251 | 251 |
| Tomato — Target Spot (Tomato___Target_Spot) | 1404 | 982 | 211 | 211 |
| Tomato — Tomato Yellow Leaf Curl Virus (Tomato___Tomato_Yellow_Leaf_Curl_Virus) | 5357 | 3749 | 804 | 804 |
| Tomato — Tomato mosaic virus (Tomato___Tomato_mosaic_virus) | 373 | 261 | 56 | 56 |
| Feuille saine (Tomato___healthy) | 1591 | 1113 | 239 | 239 |
| **Total** | **54305** | **37999** | **8153** | **8153** |

Découpage stratifié, graine 42, 21 groupes de doublons exacts (MD5 global) regroupés, 0 conflit(s) d'annotation, 196 groupes de quasi-doublons signalés, 0 image(s) illisible(s).

## 3. Courbes d'apprentissage

![Courbes accuracy et loss (entraînement / validation)](../../results/training_curves.png)

Meilleure accuracy de validation : **97,01 %** (époque 45, sur 45 époques).

| Phase | Époques | Meilleure val_accuracy | Durée |
| --- | --- | --- | --- |
| Phase 1 — base gelée | 30 | 94,75 % | 498.13 min |
| Phase 2 — fine-tuning | 15 | 97,01 % | 267.25 min |

## 4. Métriques sur l'ensemble de test

| Métrique | Valeur |
| --- | --- |
| Accuracy | **97,90 %** |
| Precision (macro) | 97,26 % |
| Recall (macro) | 97,77 % |
| F1-score (macro) | **97,44 %** |
| F1-score (pondéré) | 97,91 % |
| Images de test | 8153 |
| Images mal classées | 171 (2,10 %) |

## 5. Matrice de confusion

| Réel \ Prédit | Apple — Apple scab | Apple — Black rot | Apple — Cedar apple rust | Apple — healthy | Blueberry — healthy | Cherry (including sour) — Powdery mildew | Cherry (including sour) — healthy | Corn (maize) — Cercospora leaf spot Gray leaf spot | Corn (maize) — Common rust | Corn (maize) — Northern Leaf Blight | Corn (maize) — healthy | Grape — Black rot | Grape — Esca (Black Measles) | Grape — Leaf blight (Isariopsis Leaf Spot) | Grape — healthy | Orange — Haunglongbing (Citrus greening) | Peach — Bacterial spot | Peach — healthy | Pepper, bell — Bacterial spot | Pepper, bell — healthy | Potato — Early blight | Potato — Late blight | Potato — healthy | Raspberry — healthy | Soybean — healthy | Squash — Powdery mildew | Strawberry — Leaf scorch | Strawberry — healthy | Tomato — Bacterial spot | Alternariose précoce | Mildiou | Moisissure des feuilles | Tomato — Septoria leaf spot | Tomato — Spider mites Two-spotted spider mite | Tomato — Target Spot | Tomato — Tomato Yellow Leaf Curl Virus | Tomato — Tomato mosaic virus | Feuille saine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Apple — Apple scab | 94 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple — Black rot | 0 | 93 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple — Cedar apple rust | 0 | 0 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple — healthy | 0 | 1 | 0 | 246 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Blueberry — healthy | 0 | 0 | 0 | 0 | 224 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cherry (including sour) — Powdery mildew | 0 | 0 | 0 | 0 | 0 | 158 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cherry (including sour) — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn (maize) — Cercospora leaf spot Gray leaf spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 75 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn (maize) — Common rust | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 179 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn (maize) — Northern Leaf Blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 0 | 126 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn (maize) — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 174 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape — Black rot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 171 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape — Esca (Black Measles) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 205 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape — Leaf blight (Isariopsis Leaf Spot) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 161 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 63 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Orange — Haunglongbing (Citrus greening) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 826 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Peach — Bacterial spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 344 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Peach — healthy | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pepper, bell — Bacterial spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 147 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pepper, bell — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 220 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Potato — Early blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Potato — Late blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 140 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| Potato — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Raspberry — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Soybean — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 758 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| Squash — Powdery mildew | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strawberry — Leaf scorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 166 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strawberry — healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Tomato — Bacterial spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 312 | 1 | 2 | 0 | 2 | 0 | 2 | 0 | 0 | 0 |
| Alternariose précoce | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 123 | 6 | 1 | 3 | 2 | 9 | 0 | 0 | 0 |
| Mildiou | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 271 | 5 | 0 | 2 | 3 | 0 | 0 | 0 |
| Moisissure des feuilles | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 137 | 0 | 4 | 2 | 0 | 0 | 0 |
| Tomato — Septoria leaf spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 2 | 254 | 2 | 5 | 0 | 0 | 0 |
| Tomato — Spider mites Two-spotted spider mite | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 241 | 4 | 0 | 0 | 5 |
| Tomato — Target Spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 10 | 195 | 0 | 0 | 3 |
| Tomato — Tomato Yellow Leaf Curl Virus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 2 | 6 | 0 | 785 | 1 | 0 |
| Tomato — Tomato mosaic virus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 0 |
| Feuille saine | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 239 |

![Matrice de confusion (effectifs et proportions)](../../results/confusion_matrix.png)

## 6. Résultats par classe

| Classe | Precision | Recall | F1-score | Support | Bien classées | Confiance moyenne |
| --- | --- | --- | --- | --- | --- | --- |
| Apple — Apple scab | 98,95 % | 100,00 % | 99,47 % | 94 | 94 | 98,64 % |
| Apple — Black rot | 98,94 % | 100,00 % | 99,47 % | 93 | 93 | 100,00 % |
| Apple — Cedar apple rust | 100,00 % | 100,00 % | 100,00 % | 41 | 41 | 99,93 % |
| Apple — healthy | 99,60 % | 99,60 % | 99,60 % | 247 | 246 | 99,71 % |
| Blueberry — healthy | 100,00 % | 99,56 % | 99,78 % | 225 | 224 | 99,25 % |
| Cherry (including sour) — Powdery mildew | 100,00 % | 100,00 % | 100,00 % | 158 | 158 | 99,68 % |
| Cherry (including sour) — healthy | 98,46 % | 100,00 % | 99,22 % | 128 | 128 | 100,00 % |
| Corn (maize) — Cercospora leaf spot Gray leaf spot | 77,32 % | 97,40 % | 86,21 % | 77 | 75 | 97,42 % |
| Corn (maize) — Common rust | 100,00 % | 100,00 % | 100,00 % | 179 | 179 | 99,99 % |
| Corn (maize) — Northern Leaf Blight | 98,44 % | 85,14 % | 91,30 % | 148 | 126 | 89,51 % |
| Corn (maize) — healthy | 100,00 % | 100,00 % | 100,00 % | 174 | 174 | 100,00 % |
| Grape — Black rot | 98,84 % | 96,61 % | 97,71 % | 177 | 171 | 97,60 % |
| Grape — Esca (Black Measles) | 97,16 % | 99,03 % | 98,09 % | 207 | 205 | 99,03 % |
| Grape — Leaf blight (Isariopsis Leaf Spot) | 99,38 % | 100,00 % | 99,69 % | 161 | 161 | 100,00 % |
| Grape — healthy | 100,00 % | 100,00 % | 100,00 % | 63 | 63 | 99,93 % |
| Orange — Haunglongbing (Citrus greening) | 100,00 % | 100,00 % | 100,00 % | 826 | 826 | 99,95 % |
| Peach — Bacterial spot | 100,00 % | 99,71 % | 99,85 % | 345 | 344 | 99,70 % |
| Peach — healthy | 98,15 % | 98,15 % | 98,15 % | 54 | 53 | 99,06 % |
| Pepper, bell — Bacterial spot | 98,66 % | 98,00 % | 98,33 % | 150 | 147 | 98,23 % |
| Pepper, bell — healthy | 97,35 % | 99,10 % | 98,21 % | 222 | 220 | 99,28 % |
| Potato — Early blight | 99,33 % | 98,67 % | 99,00 % | 150 | 148 | 99,71 % |
| Potato — Late blight | 98,59 % | 93,33 % | 95,89 % | 150 | 140 | 95,55 % |
| Potato — healthy | 88,24 % | 100,00 % | 93,75 % | 30 | 30 | 98,41 % |
| Raspberry — healthy | 100,00 % | 100,00 % | 100,00 % | 56 | 56 | 99,80 % |
| Soybean — healthy | 100,00 % | 99,21 % | 99,61 % | 764 | 758 | 99,53 % |
| Squash — Powdery mildew | 100,00 % | 100,00 % | 100,00 % | 275 | 275 | 100,00 % |
| Strawberry — Leaf scorch | 100,00 % | 100,00 % | 100,00 % | 166 | 166 | 99,63 % |
| Strawberry — healthy | 100,00 % | 100,00 % | 100,00 % | 68 | 68 | 100,00 % |
| Tomato — Bacterial spot | 95,12 % | 97,81 % | 96,45 % | 319 | 312 | 96,39 % |
| Alternariose précoce | 95,35 % | 82,00 % | 88,17 % | 150 | 123 | 87,40 % |
| Mildiou | 95,76 % | 94,76 % | 95,25 % | 286 | 271 | 95,54 % |
| Moisissure des feuilles | 93,20 % | 95,80 % | 94,48 % | 143 | 137 | 97,26 % |
| Tomato — Septoria leaf spot | 96,58 % | 95,49 % | 96,03 % | 266 | 254 | 93,40 % |
| Tomato — Spider mites Two-spotted spider mite | 89,26 % | 96,02 % | 92,51 % | 251 | 241 | 94,45 % |
| Tomato — Target Spot | 88,64 % | 92,42 % | 90,49 % | 211 | 195 | 92,61 % |
| Tomato — Tomato Yellow Leaf Curl Virus | 100,00 % | 97,64 % | 98,80 % | 804 | 785 | 98,56 % |
| Tomato — Tomato mosaic virus | 98,25 % | 100,00 % | 99,12 % | 56 | 56 | 99,28 % |
| Feuille saine | 96,37 % | 100,00 % | 98,15 % | 239 | 239 | 99,31 % |

![Performances par classe](../../results/figures/per_class_performance.png)

## 7. Confiance et analyse des erreurs

| Indicateur | Valeur |
| --- | --- |
| Confiance moyenne (prédictions correctes) | 98,69 % |
| Confiance moyenne (erreurs) | 71,31 % |
| Confiance médiane | 100,00 % |
| Erreurs | 171 / 8153 (2,10 %) |

| Confusion observée | Nombre de cas |
| --- | --- |
| Corn (maize) — Northern Leaf Blight → Corn (maize) — Cercospora leaf spot Gray leaf spot | 22 |
| Tomato — Tomato Yellow Leaf Curl Virus → Tomato — Bacterial spot | 10 |
| Tomato — Target Spot → Tomato — Spider mites Two-spotted spider mite | 10 |
| Alternariose précoce → Tomato — Target Spot | 9 |
| Alternariose précoce → Mildiou | 6 |

![Distribution de la confiance](../../results/figures/confidence_distribution.png)

![Exemples d'erreurs de classification](../../results/predictions/error_grid.png)

## 8. Observations et difficultés rencontrées

- **Convergence rapide de la phase 1** : dès la première époque, la validation atteint 86,59 %, ce qui confirme la qualité des caractéristiques ImageNet.
- **Le fine-tuning apporte un gain réel de +2,26 point de validation** ; son learning rate très faible (1e-5) est indispensable pour ne pas détruire les poids pré-entraînés.
- **Les classes rares restent fragiles** : « Corn (maize) — Cercospora leaf spot Gray leaf spot » obtient le F1-score le plus bas (86,21 %), alors que 19 classes dépassent 99 %.
- **Le déséquilibre pèse sur les moyennes** : l'écart entre F1 pondéré et F1 macro est de 0,47 point, trace du traitement inégal des classes rares.
- **Le modèle est sur-confiant** : la confiance médiane atteint 100,000 % ; cette quasi-certitude justifie l'affichage des probabilités plutôt qu'une classe unique.
- **Budget d'entraînement lourd** : 45 époques au total sur 4 threads CPU (≈ 765 minutes).
- **Dataset de laboratoire** : les images ont un fond uniforme ; les performances mesurées ne préjugent pas du comportement sur des photos de terrain.
- **Doublons** : 21 groupes d'images identiques ont été détectés (empreinte MD5 calculée sur l'ensemble du dataset, toutes classes confondues) et regroupés pour éviter toute fuite ; 0 conflit(s) d'annotation et 196 groupes de quasi-doublons (empreinte perceptuelle) sont signalés pour inspection.

## 9. Conclusion de l'expérience

Le modèle MobileNetV2 adapté par Transfer Learning atteint **97,90 % d'accuracy** et un **F1-score macro de 97,44 %** sur 8153 images de test indépendantes, avec un temps d'inférence d'environ 110 ms par image sur CPU. Ces résultats confirment la faisabilité d'un outil local d'aide à la reconnaissance visuelle, dans les limites du périmètre étudié.

## 10. Fichiers de référence

| Ressource | Fichier |
| --- | --- |
| Configuration complète | `models/training_config.json` |
| Historique des époques | `results/metrics/training_history.json` |
| Rapport de classification | `results/metrics/classification_report.txt` |
| Métriques détaillées | `results/metrics.json` |
| Prédictions du test | `results/predictions/predictions_test.csv` |
| Erreurs détaillées | `results/predictions/erreurs.csv` |
