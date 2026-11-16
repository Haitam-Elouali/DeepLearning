# Résultats de l'évaluation — ensemble de test

- Modèle : `best_model.keras`
- Date : 16/11/2026 22:47
- Images de test : 8153

## Métriques globales

| Métrique | Valeur |
| --- | --- |
| Accuracy | 0.9790 |
| Precision (macro) | 0.9726 |
| Recall (macro) | 0.9777 |
| F1-score (macro) | 0.9744 |
| F1-score (pondéré) | 0.9791 |
| Images mal classées | 171 / 8153 |

## Performances par classe

| Classe | Precision | Recall | F1 | Support | Bien classées |
| --- | --- | --- | --- | --- | --- |
| Apple___Apple_scab | 0.9895 | 1.0000 | 0.9947 | 94 | 94 |
| Apple___Black_rot | 0.9894 | 1.0000 | 0.9947 | 93 | 93 |
| Apple___Cedar_apple_rust | 1.0000 | 1.0000 | 1.0000 | 41 | 41 |
| Apple___healthy | 0.9960 | 0.9960 | 0.9960 | 247 | 246 |
| Blueberry___healthy | 1.0000 | 0.9956 | 0.9978 | 225 | 224 |
| Cherry_(including_sour)___Powdery_mildew | 1.0000 | 1.0000 | 1.0000 | 158 | 158 |
| Cherry_(including_sour)___healthy | 0.9846 | 1.0000 | 0.9922 | 128 | 128 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.7732 | 0.9740 | 0.8621 | 77 | 75 |
| Corn_(maize)___Common_rust_ | 1.0000 | 1.0000 | 1.0000 | 179 | 179 |
| Corn_(maize)___Northern_Leaf_Blight | 0.9844 | 0.8514 | 0.9130 | 148 | 126 |
| Corn_(maize)___healthy | 1.0000 | 1.0000 | 1.0000 | 174 | 174 |
| Grape___Black_rot | 0.9884 | 0.9661 | 0.9771 | 177 | 171 |
| Grape___Esca_(Black_Measles) | 0.9716 | 0.9903 | 0.9809 | 207 | 205 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0.9938 | 1.0000 | 0.9969 | 161 | 161 |
| Grape___healthy | 1.0000 | 1.0000 | 1.0000 | 63 | 63 |
| Orange___Haunglongbing_(Citrus_greening) | 1.0000 | 1.0000 | 1.0000 | 826 | 826 |
| Peach___Bacterial_spot | 1.0000 | 0.9971 | 0.9985 | 345 | 344 |
| Peach___healthy | 0.9815 | 0.9815 | 0.9815 | 54 | 53 |
| Pepper,_bell___Bacterial_spot | 0.9866 | 0.9800 | 0.9833 | 150 | 147 |
| Pepper,_bell___healthy | 0.9735 | 0.9910 | 0.9821 | 222 | 220 |
| Potato___Early_blight | 0.9933 | 0.9867 | 0.9900 | 150 | 148 |
| Potato___Late_blight | 0.9859 | 0.9333 | 0.9589 | 150 | 140 |
| Potato___healthy | 0.8824 | 1.0000 | 0.9375 | 30 | 30 |
| Raspberry___healthy | 1.0000 | 1.0000 | 1.0000 | 56 | 56 |
| Soybean___healthy | 1.0000 | 0.9921 | 0.9961 | 764 | 758 |
| Squash___Powdery_mildew | 1.0000 | 1.0000 | 1.0000 | 275 | 275 |
| Strawberry___Leaf_scorch | 1.0000 | 1.0000 | 1.0000 | 166 | 166 |
| Strawberry___healthy | 1.0000 | 1.0000 | 1.0000 | 68 | 68 |
| Tomato___Bacterial_spot | 0.9512 | 0.9781 | 0.9645 | 319 | 312 |
| Tomato___Early_blight | 0.9535 | 0.8200 | 0.8817 | 150 | 123 |
| Tomato___Late_blight | 0.9576 | 0.9476 | 0.9525 | 286 | 271 |
| Tomato___Leaf_Mold | 0.9320 | 0.9580 | 0.9448 | 143 | 137 |
| Tomato___Septoria_leaf_spot | 0.9658 | 0.9549 | 0.9603 | 266 | 254 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.8926 | 0.9602 | 0.9251 | 251 | 241 |
| Tomato___Target_Spot | 0.8864 | 0.9242 | 0.9049 | 211 | 195 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 1.0000 | 0.9764 | 0.9880 | 804 | 785 |
| Tomato___Tomato_mosaic_virus | 0.9825 | 1.0000 | 0.9912 | 56 | 56 |
| Tomato___healthy | 0.9637 | 1.0000 | 0.9815 | 239 | 239 |

## Matrice de confusion

| Réel \ Prédit | Apple___Apple_scab | Apple___Black_rot | Apple___Cedar_apple_rust | Apple___healthy | Blueberry___healthy | Cherry_(including_sour)___Powdery_mildew | Cherry_(including_sour)___healthy | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Common_rust_ | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___healthy | Grape___Black_rot | Grape___Esca_(Black_Measles) | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | Grape___healthy | Orange___Haunglongbing_(Citrus_greening) | Peach___Bacterial_spot | Peach___healthy | Pepper,_bell___Bacterial_spot | Pepper,_bell___healthy | Potato___Early_blight | Potato___Late_blight | Potato___healthy | Raspberry___healthy | Soybean___healthy | Squash___Powdery_mildew | Strawberry___Leaf_scorch | Strawberry___healthy | Tomato___Bacterial_spot | Tomato___Early_blight | Tomato___Late_blight | Tomato___Leaf_Mold | Tomato___Septoria_leaf_spot | Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Target_Spot | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato___Tomato_mosaic_virus | Tomato___healthy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Apple___Apple_scab | 94 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple___Black_rot | 0 | 93 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple___Cedar_apple_rust | 0 | 0 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple___healthy | 0 | 1 | 0 | 246 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Blueberry___healthy | 0 | 0 | 0 | 0 | 224 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cherry_(including_sour)___Powdery_mildew | 0 | 0 | 0 | 0 | 0 | 158 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cherry_(including_sour)___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 75 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___Common_rust_ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 179 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___Northern_Leaf_Blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 0 | 126 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 174 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___Black_rot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 171 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___Esca_(Black_Measles) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 205 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 161 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 63 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Orange___Haunglongbing_(Citrus_greening) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 826 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Peach___Bacterial_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 344 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Peach___healthy | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pepper,_bell___Bacterial_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 147 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pepper,_bell___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 220 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Potato___Early_blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 148 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Potato___Late_blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 140 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| Potato___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Raspberry___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Soybean___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 758 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| Squash___Powdery_mildew | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strawberry___Leaf_scorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 166 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strawberry___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Tomato___Bacterial_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 312 | 1 | 2 | 0 | 2 | 0 | 2 | 0 | 0 | 0 |
| Tomato___Early_blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 123 | 6 | 1 | 3 | 2 | 9 | 0 | 0 | 0 |
| Tomato___Late_blight | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 271 | 5 | 0 | 2 | 3 | 0 | 0 | 0 |
| Tomato___Leaf_Mold | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 137 | 0 | 4 | 2 | 0 | 0 | 0 |
| Tomato___Septoria_leaf_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 2 | 254 | 2 | 5 | 0 | 0 | 0 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 241 | 4 | 0 | 0 | 5 |
| Tomato___Target_Spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 10 | 195 | 0 | 0 | 3 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 2 | 6 | 0 | 785 | 1 | 0 |
| Tomato___Tomato_mosaic_virus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 0 |
| Tomato___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 239 |

## Analyse des erreurs

- Confiance moyenne sur les prédictions correctes : 0.9869
- Confiance moyenne sur les erreurs : 0.7131
- Corn_(maize)___Northern_Leaf_Blight → Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot : 22 cas
- Tomato___Tomato_Yellow_Leaf_Curl_Virus → Tomato___Bacterial_spot : 10 cas
- Tomato___Target_Spot → Tomato___Spider_mites Two-spotted_spider_mite : 10 cas
- Tomato___Early_blight → Tomato___Target_Spot : 9 cas
- Tomato___Early_blight → Tomato___Late_blight : 6 cas
