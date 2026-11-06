# Résultats de l'évaluation — ensemble de test

- Modèle : `best_model.keras`
- Date : 06/11/2026 13:17
- Images de test : 820

## Métriques globales

| Métrique | Valeur |
| --- | --- |
| Accuracy | 0.9585 |
| Precision (macro) | 0.9547 |
| Recall (macro) | 0.9525 |
| F1-score (macro) | 0.9534 |
| F1-score (pondéré) | 0.9582 |
| Images mal classées | 34 / 820 |

## Performances par classe

| Classe | Precision | Recall | F1 | Support | Bien classées |
| --- | --- | --- | --- | --- | --- |
| Tomato___Early_blight | 0.9366 | 0.8867 | 0.9110 | 150 | 133 |
| Tomato___Late_blight | 0.9615 | 0.9582 | 0.9599 | 287 | 275 |
| Tomato___Leaf_Mold | 0.9452 | 0.9650 | 0.9550 | 143 | 138 |
| Tomato___healthy | 0.9756 | 1.0000 | 0.9877 | 240 | 240 |

## Matrice de confusion

| Réel \ Prédit | Tomato___Early_blight | Tomato___Late_blight | Tomato___Leaf_Mold | Tomato___healthy |
| --- | --- | --- | --- | --- |
| Tomato___Early_blight | 133 | 9 | 4 | 4 |
| Tomato___Late_blight | 6 | 275 | 4 | 2 |
| Tomato___Leaf_Mold | 3 | 2 | 138 | 0 |
| Tomato___healthy | 0 | 0 | 0 | 240 |

## Analyse des erreurs

- Confiance moyenne sur les prédictions correctes : 0.9752
- Confiance moyenne sur les erreurs : 0.7023
- Tomato___Early_blight → Tomato___Late_blight : 9 cas
- Tomato___Late_blight → Tomato___Early_blight : 6 cas
- Tomato___Early_blight → Tomato___healthy : 4 cas
- Tomato___Late_blight → Tomato___Leaf_Mold : 4 cas
- Tomato___Early_blight → Tomato___Leaf_Mold : 4 cas
