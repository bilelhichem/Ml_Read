```markdown
## Data Preprocessing en Python

### 1. Introduction

Le prétraitement des données est une étape essentielle en Data Science. Ce processus permet de nettoyer, transformer et préparer les données pour une analyse plus efficace et précise.

### 2. Étapes nécessaires

#### 2.1 Importation des bibliothèques

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

#### 2.2 Lecture du jeu de données

```python
df = pd.read_csv("./student/train.csv")
df.head()
```

#### 2.3 Vérification de la cohérence des données

```python
df.isnull().sum()  # Nombre de valeurs manquantes
df.isnull().sum() / df.shape[0] * 100  # Pourcentage de valeurs manquantes
```

**Gestion des valeurs manquantes selon les modèles :**

- **XGBoost** : Gère directement les valeurs manquantes. Il apprend le meilleur chemin pour les données manquantes lors de la construction des arbres.
- **Random Forest** : Ne supporte pas les valeurs manquantes directement. Il faut les imputer (moyenne, médiane...) avant d'entraîner le modèle.
- **Linear Regression** : Ne peut pas gérer les valeurs manquantes et nécessite une imputation préalable.

#### 2.4 Gestion des valeurs manquantes et aberrantes

##### Détection des valeurs aberrantes avec un Boxplot

```python
sns.boxplot(x=df['colonne'])
```

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

#### 2.5 Conversion et vérification des types de données

```python
df.dtypes
df['colonne'] = df['colonne'].astype(int)  # Conversion en entier
```

#### 2.6 Analyse de la corrélation

```python
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

##### Mutual Information

La **mutual information** détecte les relations complexes entre les variables et aide à identifier les features les plus informatives pour la variable cible.

```python
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(df.drop('cible', axis=1), df['cible'])
pd.Series(mi_scores, index=df.drop('cible', axis=1).columns).sort_values(ascending=False)
```

#### 2.7 Encodage des variables catégorielles

- **Label Encoding** : à utiliser pour les variables ordinales (ordre logique entre les catégories).
- **One Hot Encoding** : préférable pour les variables nominales, car chaque catégorie devient une colonne binaire.

```python
df = pd.get_dummies(df, columns=['categorie'], drop_first=True)
```

#### 2.8 Normalisation et standardisation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['colonne']])
```

#### 2.9 Correction de la Skewness (asymétrie)

La **skewness** permet de comprendre comment les données sont réparties et si des ajustements sont nécessaires.

- La transformation **Yeo-Johnson** aide à rendre les données plus symétriques.

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df['colonne_transforme'] = pt.fit_transform(df[['colonne']])
```

### 3. Manipulation des Données avec Pandas

#### 3.1 Création et manipulation de DataFrame

```python
# Création d'un DataFrame
df = pd.DataFrame({'Nom': ['Alice', 'Bob'], 'Age': [25, 30]})
df.columns = ['ID', 'Nom', 'Age']  # Renommer les colonnes
```

#### 3.2 Opérations courantes

```python
df.head()  # Afficher les premières lignes
df.info()  # Informations sur le DataFrame
df.describe()  # Statistiques descriptives
df.dropna(inplace=True)  # Suppression des valeurs NaN
df.fillna(value=0, inplace=True)  # Remplacement des valeurs NaN
df.drop_duplicates(subset=['Nom'], keep='first', inplace=True)  # Suppression des doublons
df = pd.concat([df1, df2], ignore_index=True)  # Fusionner deux DataFrames
```

### 4. NumPy en Python

#### 4.1 Manipulation des tableaux NumPy

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])  # Création d'un tableau NumPy
arr = np.array([[1, 2, 3], [4, 5, 6]])  # Tableau 2D
arr[1:4]  # Extraction des éléments
np.random.shuffle(arr)  # Mélanger les valeurs
```

### 5. Large Language Models (LLMs)

#### 5.1 Niveaux d'utilisation des LLMs

- **Niveau 1 - Prompt Engineering** : Rédaction de prompts efficaces
- **Niveau 2 - Model Fine-tuning** : Adaptation d'un modèle pré-entraîné
- **Niveau 3 - Build Your Own** : Entraînement d'un modèle à partir de zéro
  
## Modèle

### 1) Fonction de coût et erreur

La fonction que tu montres sur l’image est la racine de l’erreur quadratique moyenne (RMSE - Root Mean Squared Error) :  
Utilisée pour évaluer la performance du modèle après l’entraînement.

```python
# Erreur quadratique moyenne (RMSE)
err = (1 / m) * sum((f_hat(x_i) - y_i) ** 2 for i in range(1, m + 1))
```

Alors que la fonction de coût que tu avais mentionnée précédemment est la fonction de coût de la régression linéaire (MSE - Mean Squared Error) :  
Utilisée pour entraîner le modèle et ajuster \(\theta_0, \theta_1\) en minimisant l’erreur.

```python
# Fonction de coût (MSE)
J_theta = (1 / (2 * m)) * sum((h(x_i) - y_i) ** 2 for i in range(1, m + 1))
```

### 2) Décomposition de l'erreur biais-variance (RMSE)

```python
# Erreur totale
err = (1 / m) * sum((f_hat(x_i) - y_i) ** 2 for i in range(1, m + 1))
```

On décompose l'erreur en trois termes :  

- **Biais** (\(Bias^2(x_0)\))  
- **Variance** (\(Var[\hat{f}(x_0)]\))  
- **Bruit** (Erreur irrécupérable, qui provient du bruit dans les données)

#### Biais, Variance et Bruit

L'erreur irrécupérable est une partie de l'erreur qui ne peut pas être réduite, car elle est causée par le bruit des données.

```python
# Biais
Bias = np.mean(f_hat(x_i) - y_i)

# Variance
Variance = np.var(f_hat(x_i))

# Bruit
Bruit = "Erreur irrécupérable due au bruit des données"
```

- Le **biais** mesure l'écart entre la moyenne des prédictions du modèle et la valeur que l'on essaie de prédire.  
  Lorsque le biais est élevé, le modèle fait souvent des prédictions erronées.
  
- La **variance** d'un modèle fait référence à sa sensibilité aux variations des données.  
  Si la variance est élevée, les prédictions du modèle peuvent varier énormément en fonction des données d'entraînement.

## Évaluation des Modèles de Classification

## 1. Matrice de confusion
|                | Prédire Positif | Prédire Négatif |
|---------------|----------------|----------------|
| **Classe positive** | TP | FN |
| **Classe négative** | FP | TN |

## 2. Principales métriques
- **Taux de vrais positifs (Recall)** : \( TPR = \frac{TP}{P} \)
- **Taux de faux positifs** : \( FPR = \frac{FP}{N} \)
- **Taux de vrais négatifs (Spécificité)** : \( TNR = \frac{TN}{N} \)
- **Taux de faux négatifs** : \( FNR = \frac{FN}{P} \)

## 3. Taux d'erreur pondéré
\( \pi_P \times \text{FNR} + \pi_N \times \text{FPR} \)

## 4. Coût de classification
\( C = C_{FP} \times FP + C_{FN} \times FN \)
```

