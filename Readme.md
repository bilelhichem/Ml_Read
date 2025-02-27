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

### 2.7 Encodage des variables catégorielles

```python
df = pd.get_dummies(df, columns=['categorie'], drop_first=True)
```

### 2.8 Normalisation et standardisation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['colonne']])
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

