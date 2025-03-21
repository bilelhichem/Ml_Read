# Data Preprocessing en Python

## 1. Importation des Bibliothèques

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

## 2. Lecture du Jeu de Données

```python
df = pd.read_csv("./student/train.csv")
df.head()
```

## 3. Vérification de la Cohérence des Données

### 3.1 Détection des Valeurs Manquantes

```python
df.isnull().sum()  # Nombre de valeurs manquantes
df.isnull().sum() / df.shape[0] * 100  # Pourcentage de valeurs manquantes
```

### 3.2 Gestion des Valeurs Manquantes selon les Modèles

- **XGBoost** : Gère directement les valeurs manquantes.
- **Random Forest** : Nécessite une imputation préalable (moyenne, médiane, etc.).
- **Régression Linéaire** : Nécessite une imputation préalable.

## 4. Gestion des Valeurs Manquantes et Aberrantes

### 4.1 Détection des Valeurs Aberrantes avec un Boxplot

```python
sns.boxplot(x=df['colonne'])
```

### 4.2 Méthodes de Détection des Valeurs Aberrantes

#### Intervalle Interquartile (IQR)

```python
Q1 = df['colonne'].quantile(0.25)
Q3 = df['colonne'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['colonne'] < (Q1 - 1.5 * IQR)) | (df['colonne'] > (Q3 + 1.5 * IQR))]
```

#### Z-score

```python
from scipy import stats
df['z_score'] = np.abs(stats.zscore(df['colonne']))
outliers = df[df['z_score'] > 3]
```

## 5. Conversion et Vérification des Types de Données

```python
df.dtypes
df['colonne'] = df['colonne'].astype(int)  # Conversion en entier
```

## 6. Analyse de la Corrélation

```python
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

### 6.1 Mutual Information

```python
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(df.drop('cible', axis=1), df['cible'])
pd.Series(mi_scores, index=df.drop('cible', axis=1).columns).sort_values(ascending=False)
```

## 7. Encodage des Variables Catégorielles

- **Label Encoding** : Pour les variables ordinales.
- **One-Hot Encoding** : Pour les variables nominales.

```python
df = pd.get_dummies(df, columns=['categorie'], drop_first=True)
```

## 8. Normalisation et Standardisation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['colonne']])
```

## 9. Correction de la Skewness (Asymétrie)

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df['colonne_transforme'] = pt.fit_transform(df[['colonne']])
```

## 10. Manipulation des Données avec Pandas

### 10.1 Création et Manipulation de DataFrame

```python
df = pd.DataFrame({'Nom': ['Alice', 'Bob'], 'Age': [25, 30]})
df.columns = ['ID', 'Nom', 'Age']  # Renommer les colonnes
```

### 10.2 Opérations Courantes

```python
df.head()  # Afficher les premières lignes
df.info()  # Informations sur le DataFrame
df.describe()  # Statistiques descriptives
df.dropna(inplace=True)  # Suppression des valeurs NaN
df.fillna(value=0, inplace=True)  # Remplacement des valeurs NaN
df.drop_duplicates(subset=['Nom'], keep='first', inplace=True)  # Suppression des doublons
df = pd.concat([df1, df2], ignore_index=True)  # Fusionner deux DataFrames
```

## 11. NumPy en Python

### 11.1 Manipulation des Tableaux NumPy

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])  # Création d'un tableau NumPy
arr = np.array([[1, 2, 3], [4, 5, 6]])  # Tableau 2D
arr[1:4]  # Extraction des éléments
np.random.shuffle(arr)  # Mélanger les valeurs
```



## 13. Modèle

### 13.1 Fonction de Coût et Erreur

#### Erreur Quadratique Moyenne (RMSE)

\[
\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (f_{\text{hat}}(x_i) - y_i)^2}
\]

#### Fonction de Coût de la Régression Linéaire (MSE)

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h(x_i) - y_i)^2
\]

### 13.2 Décomposition de l'Erreur Biais-Variance

\[
\text{Erreur Totale} = \text{Biais}^2 + \text{Variance} + \text{Bruit}
\]

- **Biais** : Mesure l'écart entre la moyenne des prédictions du modèle et la valeur réelle.
- **Variance** : Représente la sensibilité du modèle aux variations des données d'entraînement.
- **Bruit** : Erreur irrécupérable due aux imperfections des données.

## 14. Évaluation des Modèles de Classification

### 14.1 Matrice de Confusion

|                | Prédire Positif | Prédire Négatif |
|----------------|-----------------|-----------------|
| **Classe positive** | TP              | FN              |
| **Classe négative** | FP              | TN              |

### 14.2 Principales Métriques

- **Taux de vrais positifs (Recall)** : \( TPR = \frac{TP}{P} \)
- **Taux de faux positifs** : \( FPR = \frac{FP}{N} \)
- **Taux de vrais négatifs (Spécificité)** : \( TNR = \frac{TN}{N} \)
- **Taux de faux négatifs** : \( FNR = \frac{FN}{P} \)

### 14.3 Taux d'Erreur Pondéré

\[
\pi_P \times \text{FNR} + \pi_N \times \text{FPR}
\]

### 14.4 Coût de Classification

\[
C = C_{FP} \times FP + C_{FN} \times FN
\]

## 15. Techniques Avancées

### 15.1 Validation Croisée

La validation croisée divise les données en K parties (ou "folds"), puis entraîne et teste le modèle K fois.

\[
\text{Erreur} = \frac{1}{K} \sum_{i=1}^{K} \text{err}(i)
\]

### 15.2 Bootstrap

Le **Bootstrap** crée plusieurs sous-échantillons en tirant aléatoirement des données avec **remise**.

## 16. Sélection des Variables

La sélection des variables vise à choisir les caractéristiques les plus pertinentes pour prédire la variable cible. Il existe plusieurs méthodes :

1. **Méthodes de Score** : Attribuent un score à chaque variable pour évaluer sa pertinence.
2. **Méthodes de Sous-Ensembles** : Sélectionnent un sous-ensemble réduit de variables pour améliorer la performance du modèle.
3. **Méthode Wrapper** : Évalue les sous-ensembles de variables en fonction de la performance du modèle.

## 17. Régression Logistique

La régression logistique est un algorithme d'apprentissage supervisé utilisé principalement pour la classification binaire (deux classes : 0 ou 1)

La régression logistique utilise une transformation sigmoïde pour convertir une combinaison linéaire des variables d’entrée en une probabilité entre 0 et 1.

h(x) = 1/1+exp(-z)

## 18. Support Vector Machine (SVM)
 
un algorithme d’apprentissage supervisé utilisé pour la classification et la régression. Son but principal est de trouver une hyperplan optimal qui sépare les données en classes distinctes avec une marge maximale

Un hyperplan avec une grande marge réduit le risque de mauvaise généralisation


---

## 12. Large Language Models (LLMs)

### 12.1 Niveaux d'Utilisation des LLMs

- **Niveau 1 - Prompt Engineering** : Rédaction de prompts efficaces.
- **Niveau 2 - Model Fine-tuning** : Adaptation d'un modèle pré-entraîné.
- **Niveau 3 - Build Your Own** : Entraînement d'un modèle à partir de zéro.