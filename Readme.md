# Data Preprocessing en Python

## 1. Introduction

Le prétraitement des données est une étape essentielle en Data Science. Ce processus permet de nettoyer, transformer et préparer les données pour une analyse plus efficace et précise.

## 2. Étapes nécessaires

### 2.1 Importation des bibliothèques

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### 2.2 Lecture du jeu de données

```python
df = pd.read_csv("./student/train.csv")
df.head()
```

### 2.3 Vérification de la cohérence des données

```python
df.isnull().sum()  # Nombre de valeurs manquantes
df.isnull().sum() / df.shape[0] * 100  # Pourcentage de valeurs manquantes
```

**Gestion des valeurs manquantes selon les modèles :**
- **XGBoost** : Gère directement les valeurs manquantes. Il apprend le meilleur chemin pour les données manquantes lors de la construction des arbres.
- **Random Forest** : Ne supporte pas les valeurs manquantes directement. Il faut les imputer (moyenne, médiane...) avant d'entraîner le modèle.
- **Linear Regression** : Ne peut pas gérer les valeurs manquantes et nécessite une imputation préalable.

### 2.4 Gestion des valeurs manquantes et aberrantes

#### Détection des valeurs aberrantes avec un Boxplot

```python
sns.boxplot(x=df['colonne'])
```

![Boxplot Image](https://github.com/user-attachments/assets/4b4403fe-6576-4921-b931-9b4659ddc9e1)

#### Méthodes de détection des valeurs aberrantes

- **Intervalle interquartile (IQR)**

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

### 2.5 Conversion et vérification des types de données

```python
df.dtypes
df['colonne'] = df['colonne'].astype(int)  # Conversion en entier
```

### 2.6 Analyse de la corrélation

```python
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

#### Mutual Information

La **mutual information** détecte les relations complexes entre les variables et aide à identifier les features les plus informatives pour la variable cible.

```python
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(df.drop('cible', axis=1), df['cible'])
pd.Series(mi_scores, index=df.drop('cible', axis=1).columns).sort_values(ascending=False)
```

### 2.7 Encodage des variables catégorielles

- **Label Encoding** : à utiliser pour les variables ordinales (ordre logique entre les catégories).
- **One Hot Encoding** : préférable pour les variables nominales, car chaque catégorie devient une colonne binaire.

```python
df = pd.get_dummies(df, columns=['categorie'], drop_first=True)
```

### 2.8 Normalisation et standardisation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['colonne']])
```

### 2.9 Correction de la Skewness (asymétrie)

La **skewness** permet de comprendre comment les données sont réparties et si des ajustements sont nécessaires.

- La transformation **Yeo-Johnson** aide à rendre les données plus symétriques.

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df['colonne_transforme'] = pt.fit_transform(df[['colonne']])
```

## 3. Manipulation des Données avec Pandas

### 3.1 Création et manipulation de DataFrame

```python
# Création d'un DataFrame
df = pd.DataFrame({'Nom': ['Alice', 'Bob'], 'Age': [25, 30]})
df.columns = ['ID', 'Nom', 'Age']  # Renommer les colonnes
```

### 3.2 Opérations courantes

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

### 4.1 Manipulation des tableaux NumPy

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])  # Création d'un tableau NumPy
arr = np.array([[1, 2, 3], [4, 5, 6]])  # Tableau 2D
arr[1:4]  # Extraction des éléments
np.random.shuffle(arr)  # Mélanger les valeurs
```

## 5. Large Language Models (LLMs)

### 5.1 Niveaux d'utilisation des LLMs

- **Niveau 1 - Prompt Engineering** : Rédaction de prompts efficaces
- **Niveau 2 - Model Fine-tuning** : Adaptation d'un modèle pré-entraîné
- **Niveau 3 - Build Your Own** : Entraînement d'un modèle à partir de zéro
Bien sûr, voici la notation mathématique pour les différentes fonctions et concepts que tu as mentionnés, en utilisant une notation plus mathématique au lieu de Python.

## Model

### 1) **Fonction de coût (MSE - Mean Squared Error)**

La fonction de coût est utilisée pour l'entraînement du modèle, et elle est donnée par :

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h(\mathbf{x}_i) - y_i \right)^2
\]

Où :  
- \(m\) est le nombre d'exemples d'entraînement  
- \(h(\mathbf{x}_i)\) est la prédiction du modèle pour l'exemple \(\mathbf{x}_i\)  
- \(y_i\) est la valeur réelle de l'exemple d'entraînement

### 2) **Erreur quadratique moyenne (RMSE - Root Mean Squared Error)**

L'erreur quadratique moyenne (RMSE) est utilisée pour évaluer la performance du modèle après l'entraînement, et elle est donnée par :

\[
\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( f(\mathbf{x}_i) - y_i \right)^2}
\]

Où :  
- \(f(\mathbf{x}_i)\) est la prédiction du modèle pour l'exemple \(\mathbf{x}_i\)  
- \(y_i\) est la valeur réelle de l'exemple d'entraînement

### 3) **Décomposition de l'erreur en biais, variance et bruit**

L'erreur totale peut être décomposée en trois composants :

\[
\text{Erreur totale} = \text{Biais}^2 + \text{Variance} + \text{Bruit}
\]

#### a) **Biais**

Le biais est la différence entre la moyenne des prédictions du modèle et la valeur réelle :

\[
\text{Biais} = \mathbb{E}[f(\mathbf{x})] - y
\]

Où :  
- \(\mathbb{E}[f(\mathbf{x})]\) est la moyenne des prédictions du modèle  
- \(y\) est la vraie valeur

#### b) **Variance**

La variance mesure la variabilité des prédictions du modèle en fonction des différents sous-ensembles de données d'entraînement. Elle est donnée par :

\[
\text{Variance} = \mathbb{E}[(f(\mathbf{x}) - \mathbb{E}[f(\mathbf{x})])^2]
\]

Où :  
- \(\mathbb{E}[f(\mathbf{x})]\) est la moyenne des prédictions du modèle  
- \(f(\mathbf{x})\) est la prédiction du modèle

#### c) **Bruit (Erreur irrécupérable)**

Le bruit est une erreur due aux variations aléatoires et aux imperfections dans les données. Il est irrécupérable et ne peut pas être réduit par l'entraînement du modèle :

\[
\text{Bruit} = \text{Erreur irrécupérable due aux variations des données}
\]

Cela représente une partie de l'erreur qui est causée par des facteurs externes ou aléatoires dans les données et ne peut pas être éliminée par un modèle.

---

En résumé, la décomposition complète de l'erreur totale pour un modèle d'apprentissage automatique est donnée par :

\[
\text{Erreur totale} = \underbrace{\text{Biais}^2}_{\text{Erreur systématique}} + \underbrace{\text{Variance}}_{\text{Erreur due à la sensibilité du modèle aux données}} + \underbrace{\text{Bruit}}_{\text{Erreur irréductible due aux données}}
\]