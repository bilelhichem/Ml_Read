# Data Preprocessing en Python

## 2. Étapes nécessaires

### 2.1 Importation des Bibliothèques

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### 2.2 Lecture du Jeu de Données

```python
df = pd.read_csv("./student/train.csv")
df.head()
```

### 2.3 Vérification de la Cohérence des Données

```python
df.isnull().sum()  # Nombre de valeurs manquantes
df.isnull().sum() / df.shape[0] * 100  # Pourcentage de valeurs manquantes
```

**Gestion des valeurs manquantes selon les modèles :**
 
- **XGBoost** : Gère directement les valeurs manquantes en apprenant le meilleur chemin pour les données manquantes lors de la construction des arbres.
- **Random Forest** : Ne supporte pas les valeurs manquantes directement. Il faut les imputer (moyenne, médiane...) avant d'entraîner le modèle.
- **Régression Linéaire** : Ne peut pas gérer les valeurs manquantes, une imputation préalable est donc nécessaire.

### 2.4 Gestion des Valeurs Manquantes et Aberrantes

#### Détection des Valeurs Aberrantes avec un Boxplot

```python
sns.boxplot(x=df['colonne'])
```

#### Méthodes de Détection des Valeurs Aberrantes

- **Intervalle Interquartile (IQR)**

```python
Q1 = df['colonne'].quantile(0.25)
Q3 = df['colonne'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['colonne'] < (Q1 - 1.5 * IQR)) | (df['colonne'] > (Q3 + 1.5 * IQR))]
```

- **Z-score**

```python
from scipy import stats
df['z_score'] = np.abs(stats.zscore(df['colonne']))
outliers = df[df['z_score'] > 3]
```

### 2.5 Conversion et Vérification des Types de Données

```python
df.dtypes
df['colonne'] = df['colonne'].astype(int)  # Conversion en entier
```

### 2.6 Analyse de la Corrélation

```python
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

#### Mutual Information

La **mutual information** permet de détecter des relations complexes entre les variables et aide à identifier les variables les plus informatives pour la variable cible.

```python
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(df.drop('cible', axis=1), df['cible'])
pd.Series(mi_scores, index=df.drop('cible', axis=1).columns).sort_values(ascending=False)
```

### 2.7 Encodage des Variables Catégorielles

- **Label Encoding** : Utilisé pour les variables ordinales (présentant un ordre logique entre les catégories).
- **One-Hot Encoding** : Préféré pour les variables nominales, chaque catégorie devient une colonne binaire.

```python
df = pd.get_dummies(df, columns=['categorie'], drop_first=True)
```

### 2.8 Normalisation et Standardisation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['colonne']])
```

### 2.9 Correction de la Skewness (Asymétrie)

La **skewness** permet de comprendre comment les données sont réparties et si des ajustements sont nécessaires. La transformation **Yeo-Johnson** aide à rendre les données plus symétriques.

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df['colonne_transforme'] = pt.fit_transform(df[['colonne']])
```

## 3. Manipulation des Données avec Pandas

### 3.1 Création et Manipulation de DataFrame

```python
# Création d'un DataFrame
df = pd.DataFrame({'Nom': ['Alice', 'Bob'], 'Age': [25, 30]})
df.columns = ['ID', 'Nom', 'Age']  # Renommer les colonnes
```

### 3.2 Opérations Courantes

```python
df.head()  # Afficher les premières lignes
df.info()  # Informations sur le DataFrame
df.describe()  # Statistiques descriptives
df.dropna(inplace=True)  # Suppression des valeurs NaN
df.fillna(value=0, inplace=True)  # Remplacement des valeurs NaN
df.drop_duplicates(subset=['Nom'], keep='first', inplace=True)  # Suppression des doublons
df = pd.concat([df1, df2], ignore_index=True)  # Fusionner deux DataFrames
```

## 4. NumPy en Python

### 4.1 Manipulation des Tableaux NumPy

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])  # Création d'un tableau NumPy
arr = np.array([[1, 2, 3], [4, 5, 6]])  # Tableau 2D
arr[1:4]  # Extraction des éléments
np.random.shuffle(arr)  # Mélanger les valeurs
```

## 5. Large Language Models (LLMs)

### 5.1 Niveaux d'Utilisation des LLMs

- **Niveau 1 - Prompt Engineering** : Rédaction de prompts efficaces.
- **Niveau 2 - Model Fine-tuning** : Adaptation d'un modèle pré-entraîné.
- **Niveau 3 - Build Your Own** : Entraînement d'un modèle à partir de zéro.

---

## 6. Modèle

### 6.1 Fonction de Coût et Erreur

#### Erreur Quadratique Moyenne (RMSE)

La fonction RMSE permet d'évaluer la performance du modèle après l'entraînement.

```python
# Erreur quadratique moyenne (RMSE)
err = (1 / m) * sum((f_hat(x_i) - y_i) ** 2 for i in range(1, m + 1))
```

#### Fonction de Coût de la Régression Linéaire (MSE)

La fonction de coût pour la régression linéaire est utilisée pour entraîner le modèle et ajuster les paramètres (\(\theta_0, \theta_1\)).

```python
# Fonction de coût (MSE)
J_theta = (1 / (2 * m)) * sum((h(x_i) - y_i) ** 2 for i in range(1, m + 1))
```

### 6.2 Décomposition de l'Erreur Biais-Variance

L'erreur totale peut être décomposée en trois termes : Biais, Variance et Bruit.

```python
# Biais
Bias = np.mean(f_hat(x_i) - y_i)

# Variance
Variance = np.var(f_hat(x_i))

# Bruit
Bruit = "Erreur irrécupérable due au bruit des données"
```

#### Interprétation

- **Biais** mesure l'écart entre la moyenne des prédictions du modèle et la valeur réelle.
- **Variance** représente la sensibilité du modèle aux variations des données d'entraînement.
- **Bruit** est l'erreur irrécupérable due aux imperfections des données.

---

## 7. Évaluation des Modèles de Classification

### 7.1 Matrice de Confusion

|                | Prédire Positif | Prédire Négatif |
|----------------|-----------------|-----------------|
| **Classe positive** | TP              | FN              |
| **Classe négative** | FP              | TN              |

### 7.2 Principales Métriques

- **Taux de vrais positifs (Recall)** : \( TPR = \frac{TP}{P} \)
- **Taux de faux positifs** : \( FPR = \frac{FP}{N} \)
- **Taux de vrais négatifs (Spécificité)** : \( TNR = \frac{TN}{N} \)
- **Taux de faux négatifs** : \( FNR = \frac{FN}{P} \)

### 7.3 Taux d'Erreur Pondéré

```python
\pi_P \times \text{FNR} + \pi_N \times \text{FPR}
```

### 7.4 Coût de Classification

```python
C = C_{FP} \times FP + C_{FN} \times FN
```

---

## 8. Techniques Avancées

### 8.1 Validation Croisée

La **validation croisée** permet de réutiliser les données en testant le modèle sur différentes parties du dataset. Elle divise les données en **K parties** (ou "folds"), et entraîne et teste le modèle **K fois**.

```python
Erreur = \frac{1}{K} \sum_{i=1}^{K} \text{err}(i)
```

### 8.2 Bootstrap

Le **Bootstrap** crée plusieurs sous-échantillons en tirant aléatoirement des données avec **remise** (un même exemple peut être sélectionné plusieurs fois).

---

## 9. Sélection des Variables

La sélection des variables vise à choisir les caractéristiques les plus pertinentes pour prédire la variable cible. Il existe plusieurs méthodes :

1. **Méthodes de Score** : Attribuent un score à chaque variable pour évaluer sa pertinence.
2. **Méthodes de Sous-Ensembles** : Sélectionnent un sous-ensemble réduit de variables pour améliorer la performance du modèle.
3. **Méthode Wrapper** : Évalue les sous-ensembles de variables en fonction de la performance du modèle.

---
