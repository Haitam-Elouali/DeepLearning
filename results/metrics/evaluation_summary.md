# Résultats de l'évaluation — ensemble de test

- Modèle : `best_model.keras`
- Date : 15/11/2026 00:39
- Images de test : 8156

## Métriques globales

| Métrique | Valeur |
| --- | --- |
| Accuracy | 0.9706 |
| Precision (macro) | 0.9598 |
| Recall (macro) | 0.9666 |
| F1-score (macro) | 0.9617 |
| F1-score (pondéré) | 0.9705 |
| Images mal classées | 240 / 8156 |

## Performances par classe

| Classe | Precision | Recall | F1 | Support | Bien classées |
| --- | --- | --- | --- | --- | --- |
| Apple___Apple_scab | 0.9684 | 0.9787 | 0.9735 | 94 | 92 |
| Apple___Black_rot | 0.9789 | 1.0000 | 0.9894 | 93 | 93 |
| Apple___Cedar_apple_rust | 1.0000 | 1.0000 | 1.0000 | 41 | 41 |
| Apple___healthy | 0.9878 | 0.9758 | 0.9817 | 248 | 242 |
| Blueberry___healthy | 1.0000 | 0.9956 | 0.9978 | 225 | 224 |
| Cherry_(including_sour)___Powdery_mildew | 1.0000 | 1.0000 | 1.0000 | 158 | 158 |
| Cherry_(including_sour)___healthy | 0.9769 | 0.9922 | 0.9845 | 128 | 127 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.7087 | 0.9481 | 0.8111 | 77 | 73 |
| Corn_(maize)___Common_rust_ | 0.9890 | 1.0000 | 0.9944 | 179 | 179 |
| Corn_(maize)___Northern_Leaf_Blight | 0.9832 | 0.7905 | 0.8764 | 148 | 117 |
| Corn_(maize)___healthy | 0.9943 | 1.0000 | 0.9971 | 174 | 174 |
| Grape___Black_rot | 0.9887 | 0.9887 | 0.9887 | 177 | 175 |
| Grape___Esca_(Black_Measles) | 0.9856 | 0.9903 | 0.9880 | 207 | 205 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0.9877 | 0.9938 | 0.9907 | 161 | 160 |
| Grape___healthy | 0.9844 | 1.0000 | 0.9921 | 63 | 63 |
| Orange___Haunglongbing_(Citrus_greening) | 1.0000 | 1.0000 | 1.0000 | 826 | 826 |
| Peach___Bacterial_spot | 0.9971 | 0.9942 | 0.9956 | 345 | 343 |
| Peach___healthy | 0.9636 | 0.9815 | 0.9725 | 54 | 53 |
| Pepper,_bell___Bacterial_spot | 0.9732 | 0.9667 | 0.9699 | 150 | 145 |
| Pepper,_bell___healthy | 0.9776 | 0.9820 | 0.9798 | 222 | 218 |
| Potato___Early_blight | 0.9796 | 0.9600 | 0.9697 | 150 | 144 |
| Potato___Late_blight | 0.9517 | 0.9200 | 0.9356 | 150 | 138 |
| Potato___healthy | 0.8286 | 0.9667 | 0.8923 | 30 | 29 |
| Raspberry___healthy | 1.0000 | 1.0000 | 1.0000 | 56 | 56 |
| Soybean___healthy | 0.9974 | 0.9974 | 0.9974 | 764 | 762 |
| Squash___Powdery_mildew | 1.0000 | 1.0000 | 1.0000 | 275 | 275 |
| Strawberry___Leaf_scorch | 0.9940 | 1.0000 | 0.9970 | 166 | 166 |
| Strawberry___healthy | 1.0000 | 1.0000 | 1.0000 | 68 | 68 |
| Tomato___Bacterial_spot | 0.9356 | 0.9561 | 0.9457 | 319 | 305 |
| Tomato___Early_blight | 0.9381 | 0.7067 | 0.8061 | 150 | 106 |
| Tomato___Late_blight | 0.9488 | 0.9686 | 0.9586 | 287 | 278 |
| Tomato___Leaf_Mold | 0.8889 | 0.9510 | 0.9189 | 143 | 136 |
| Tomato___Septoria_leaf_spot | 0.9257 | 0.9361 | 0.9308 | 266 | 249 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.9053 | 0.9522 | 0.9282 | 251 | 239 |
| Tomato___Target_Spot | 0.8591 | 0.8957 | 0.8770 | 211 | 189 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 1.0000 | 0.9627 | 0.9810 | 804 | 774 |
| Tomato___Tomato_mosaic_virus | 0.9322 | 0.9821 | 0.9565 | 56 | 55 |
| Tomato___healthy | 0.9409 | 0.9958 | 0.9676 | 240 | 239 |

## Matrice de confusion

| Réel \ Prédit | Apple___Apple_scab | Apple___Black_rot | Apple___Cedar_apple_rust | Apple___healthy | Blueberry___healthy | Cherry_(including_sour)___Powdery_mildew | Cherry_(including_sour)___healthy | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Common_rust_ | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___healthy | Grape___Black_rot | Grape___Esca_(Black_Measles) | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | Grape___healthy | Orange___Haunglongbing_(Citrus_greening) | Peach___Bacterial_spot | Peach___healthy | Pepper,_bell___Bacterial_spot | Pepper,_bell___healthy | Potato___Early_blight | Potato___Late_blight | Potato___healthy | Raspberry___healthy | Soybean___healthy | Squash___Powdery_mildew | Strawberry___Leaf_scorch | Strawberry___healthy | Tomato___Bacterial_spot | Tomato___Early_blight | Tomato___Late_blight | Tomato___Leaf_Mold | Tomato___Septoria_leaf_spot | Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Target_Spot | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato___Tomato_mosaic_virus | Tomato___healthy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Apple___Apple_scab | 92 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple___Black_rot | 0 | 93 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple___Cedar_apple_rust | 0 | 0 | 41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Apple___healthy | 1 | 2 | 0 | 242 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Blueberry___healthy | 0 | 0 | 0 | 1 | 224 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cherry_(including_sour)___Powdery_mildew | 0 | 0 | 0 | 0 | 0 | 158 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cherry_(including_sour)___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 127 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 73 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___Common_rust_ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 179 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___Northern_Leaf_Blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 117 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Corn_(maize)___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 174 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___Black_rot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 175 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___Esca_(Black_Measles) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 205 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 160 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Grape___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 63 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Orange___Haunglongbing_(Citrus_greening) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 826 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Peach___Bacterial_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 343 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Peach___healthy | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pepper,_bell___Bacterial_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 145 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pepper,_bell___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 218 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Potato___Early_blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 144 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Potato___Late_blight | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 138 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Potato___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 29 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Raspberry___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Soybean___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 762 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Squash___Powdery_mildew | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 275 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strawberry___Leaf_scorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 166 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strawberry___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Tomato___Bacterial_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 305 | 2 | 2 | 0 | 8 | 0 | 0 | 0 | 0 | 1 |
| Tomato___Early_blight | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 106 | 7 | 4 | 6 | 2 | 14 | 0 | 1 | 1 |
| Tomato___Late_blight | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 278 | 3 | 0 | 2 | 0 | 0 | 0 | 0 |
| Tomato___Leaf_Mold | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 136 | 3 | 2 | 1 | 0 | 1 | 0 |
| Tomato___Septoria_leaf_spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 2 | 0 | 3 | 249 | 0 | 4 | 0 | 0 | 2 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 239 | 8 | 0 | 1 | 2 |
| Tomato___Target_Spot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 9 | 189 | 0 | 0 | 8 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 1 | 5 | 1 | 7 | 4 | 774 | 1 | 0 |
| Tomato___Tomato_mosaic_virus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 55 | 0 |
| Tomato___healthy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 239 |

## Analyse des erreurs

- Confiance moyenne sur les prédictions correctes : 0.9853
- Confiance moyenne sur les erreurs : 0.7444
- Corn_(maize)___Northern_Leaf_Blight → Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot : 30 cas
- Tomato___Early_blight → Tomato___Target_Spot : 14 cas
- Tomato___Tomato_Yellow_Leaf_Curl_Virus → Tomato___Bacterial_spot : 11 cas
- Tomato___Target_Spot → Tomato___Spider_mites Two-spotted_spider_mite : 9 cas
- Tomato___Target_Spot → Tomato___healthy : 8 cas
