# Résultats de l'évaluation — ensemble de test

- Modèle : `best_model.keras`
- Date : 17/11/2026 11:56
- Images de test : 8153

## Métriques globales

| Métrique | Valeur |
| --- | --- |
| Accuracy | 0.9731 |
| Precision (macro) | 0.9672 |
| Recall (macro) | 0.9668 |
| F1-score (macro) | 0.9665 |
| F1-score (pondéré) | 0.9730 |
| Images mal classées | 219 / 8153 |

## Performances par classe

| Classe | Precision | Recall | F1 | Support | Bien classées |
| --- | --- | --- | --- | --- | --- |
| Apple___Apple_scab | 0.9663 | 0.9149 | 0.9399 | 94 | 86 |
| Apple___Black_rot | 1.0000 | 0.9785 | 0.9891 | 93 | 91 |
| Apple___Cedar_apple_rust | 0.9762 | 1.0000 | 0.9880 | 41 | 41 |
| Apple___healthy | 0.9724 | 1.0000 | 0.9860 | 247 | 247 |
| Blueberry___healthy | 1.0000 | 1.0000 | 1.0000 | 225 | 225 |
| Cherry_(including_sour)___Powdery_mildew | 1.0000 | 0.9873 | 0.9936 | 158 | 156 |
| Cherry_(including_sour)___healthy | 0.9624 | 1.0000 | 0.9808 | 128 | 128 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.9091 | 0.9091 | 0.9091 | 77 | 70 |
| Corn_(maize)___Common_rust_ | 1.0000 | 1.0000 | 1.0000 | 179 | 179 |
| Corn_(maize)___Northern_Leaf_Blight | 0.9524 | 0.9459 | 0.9492 | 148 | 140 |
| Corn_(maize)___healthy | 0.9943 | 1.0000 | 0.9971 | 174 | 174 |
| Grape___Black_rot | 0.9942 | 0.9661 | 0.9799 | 177 | 171 |
| Grape___Esca_(Black_Measles) | 0.9717 | 0.9952 | 0.9833 | 207 | 206 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1.0000 | 1.0000 | 1.0000 | 161 | 161 |
| Grape___healthy | 1.0000 | 1.0000 | 1.0000 | 63 | 63 |
| Orange___Haunglongbing_(Citrus_greening) | 0.9988 | 1.0000 | 0.9994 | 826 | 826 |
| Peach___Bacterial_spot | 0.9942 | 0.9971 | 0.9957 | 345 | 344 |
| Peach___healthy | 1.0000 | 1.0000 | 1.0000 | 54 | 54 |
| Pepper,_bell___Bacterial_spot | 0.9864 | 0.9667 | 0.9764 | 150 | 145 |
| Pepper,_bell___healthy | 0.9909 | 0.9820 | 0.9864 | 222 | 218 |
| Potato___Early_blight | 1.0000 | 0.9867 | 0.9933 | 150 | 148 |
| Potato___Late_blight | 0.9793 | 0.9467 | 0.9627 | 150 | 142 |
| Potato___healthy | 0.8966 | 0.8667 | 0.8814 | 30 | 26 |
| Raspberry___healthy | 1.0000 | 1.0000 | 1.0000 | 56 | 56 |
| Soybean___healthy | 0.9961 | 0.9961 | 0.9961 | 764 | 761 |
| Squash___Powdery_mildew | 0.9964 | 1.0000 | 0.9982 | 275 | 275 |
| Strawberry___Leaf_scorch | 1.0000 | 1.0000 | 1.0000 | 166 | 166 |
| Strawberry___healthy | 1.0000 | 1.0000 | 1.0000 | 68 | 68 |
| Tomato___Bacterial_spot | 0.9475 | 0.9624 | 0.9549 | 319 | 307 |
| Tomato___Early_blight | 0.9194 | 0.7600 | 0.8321 | 150 | 114 |
| Tomato___Late_blight | 0.9468 | 0.9336 | 0.9401 | 286 | 267 |
| Tomato___Leaf_Mold | 0.8701 | 0.9371 | 0.9024 | 143 | 134 |
| Tomato___Septoria_leaf_spot | 0.9288 | 0.9323 | 0.9306 | 266 | 248 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.8782 | 0.9482 | 0.9119 | 251 | 238 |
| Tomato___Target_Spot | 0.8818 | 0.8483 | 0.8647 | 211 | 179 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.9975 | 0.9764 | 0.9868 | 804 | 785 |
| Tomato___Tomato_mosaic_virus | 0.9180 | 1.0000 | 0.9573 | 56 | 56 |
| Tomato___healthy | 0.9264 | 1.0000 | 0.9618 | 239 | 239 |

## Matrice de confusion

| Réel \ Prédit | Apple___Apple_scab | Apple___Black_rot | Apple___Cedar_apple_rust | Apple___healthy | Blueberry___healthy | Cherry_(including_sour)___Powdery_mildew | Cherry_(including_sour)___healthy | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Common_rust_ | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___healthy | Grape___Black_rot | Grape___Esca_(Black_Measles) | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | Grape___healthy | Orange___Haunglongbing_(Citrus_greening) | Peach___Bacterial_spot | Peach___healthy | Pepper,_bell___Bacterial_spot | Pepper,_bell___healthy | Potato___Early_blight | Potato___Late_blight | Potato___healthy | Raspberry___healthy | Soybean___healthy | Squash___Powdery_mildew | Strawberry___Leaf_scorch | Strawberry___healthy | Tomato___Bacterial_spot | Tomato___Early_blight | Tomato___Late_blight | Tomato___Leaf_Mold | Tomato___Septoria_leaf_spot | Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Target_Spot | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato___Tomato_mosaic_virus | Tomato___healthy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Apple___Apple_scab | 86 | 0 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple___Black_rot | 0 | 91 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple___Cedar_apple_rust | 0 | 0 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple___healthy | 0 | 0 | 0 | 247 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Blueberry___healthy | 0 | 0 | 0 | 0 | 225 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cherry_(including_sour)___Powdery_mildew | 1 | 0 | 0 | 0 | 0 | 156 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Cherry_(including_sour)___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 70 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___Common_rust_ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 179 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___Northern_Leaf_Blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 140 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 174 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___Black_rot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 171 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___Esca_(Black_Measles) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 206 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 161 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 63 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Orange___Haunglongbing_(Citrus_greening) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 826 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Peach___Bacterial_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 344 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Peach___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 54 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pepper,_bell___Bacterial_spot | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 145 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Pepper,_bell___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 218 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Potato___Early_blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Potato___Late_blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 142 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Potato___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 26 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Raspberry___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Soybean___healthy | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 761 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Squash___Powdery_mildew | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strawberry___Leaf_scorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 166 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strawberry___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Tomato___Bacterial_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 307 | 1 | 1 | 2 | 4 | 1 | 2 | 1 | 0 | 0 |
| Tomato___Early_blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 114 | 9 | 4 | 8 | 2 | 7 | 1 | 0 | 0 |
| Tomato___Late_blight | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 267 | 4 | 2 | 2 | 5 | 0 | 0 | 0 |
| Tomato___Leaf_Mold | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 134 | 1 | 5 | 2 | 0 | 0 | 0 |
| Tomato___Septoria_leaf_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 1 | 0 | 6 | 248 | 2 | 4 | 0 | 0 | 1 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 238 | 4 | 0 | 1 | 6 |
| Tomato___Target_Spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 2 | 0 | 1 | 1 | 14 | 179 | 0 | 0 | 11 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 2 | 1 | 5 | 0 | 785 | 4 | 0 |
| Tomato___Tomato_mosaic_virus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 0 |
| Tomato___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 239 |

## Analyse des erreurs

- Confiance moyenne sur les prédictions correctes : 0.9814
- Confiance moyenne sur les erreurs : 0.7234
- Tomato___Target_Spot → Tomato___Spider_mites Two-spotted_spider_mite : 14 cas
- Tomato___Target_Spot → Tomato___healthy : 11 cas
- Tomato___Early_blight → Tomato___Late_blight : 9 cas
- Tomato___Early_blight → Tomato___Septoria_leaf_spot : 8 cas
- Tomato___Tomato_Yellow_Leaf_Curl_Virus → Tomato___Bacterial_spot : 7 cas
