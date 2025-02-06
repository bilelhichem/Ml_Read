# =======================
# Data Preprocessing and Cleaning in Python
# =======================

# 1) Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 2) Lire le jeu de données
df = pd.read_csv("./student/train.csv")

# Afficher les premières lignes du dataframe
df.head()

# 3) Vérification de la cohérence des données (Sanity Check)
# Vérifier la somme des valeurs manquantes
df.isnull().sum()

# Calculer le pourcentage des valeurs manquantes
df.isnull().sum() / df.shape[0] * 100

# 4) Analyse exploratoire des données (EDA)
# Statistiques descriptives
df.describe()

# Visualiser la distribution de chaque variable numérique pour identifier les valeurs aberrantes (outliers)
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df, x=i)
    plt.show()

# Explorer les relations entre les variables avec des scatter plots
columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
           'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
           'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
           'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
           'MiscVal', 'MoSold', 'YrSold', 'SalePrice']

for i in columns:
    sns.scatterplot(data=df, x=i, y='SalePrice')
    plt.show()

# Analyser la corrélation entre les variables
df.corr()

# 5) Traitement des valeurs manquantes
# Remplir les valeurs manquantes par la moyenne de chaque colonne
df.fillna(df.mean(), inplace=True)

# Ou supprimer les lignes contenant des valeurs manquantes
df.dropna(inplace=True)

# 6) Traitement des doublons et valeurs inutiles
# Supprimer les doublons
df.drop_duplicates(inplace=True)

# 7) Normalisation des données (Min-Max)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)

# 8) Encodage des données catégorielles
df_encoded = pd.get_dummies(df, drop_first=True)

# =======================
# Introduction to Pandas in Python
# =======================

# Exemple de création d'un dataset avec pandas
mydataset = {
    'cars': ["BMW", "Volvo", "Ford"],
    'passings': [3, 7, 2]
}
myvar = pd.DataFrame(mydataset)

# Renommer les colonnes du DataFrame
df.columns = ['student_id', 'age'] 

# Référence à un index de ligne spécifique
print(myvar.loc[0])

# Afficher les deux premières lignes du DataFrame
print(df.loc[[0, 1]])

# Exemple de création d'une série pandas avec index personnalisé
a = [1, 2, 3, 4, 5]
myvar = pd.Series(a, index = ["x", "y", "z"])

# Afficher la série avec index
print(myvar)

# Lire un fichier CSV
df = pd.read_csv('./student/student-mat.csv')

# Visualiser les données
df.head()  # Afficher les premières lignes du DataFrame

df.tail()  # Afficher les dernières lignes du DataFrame

# Afficher des informations sur le DataFrame
df.info()  

# Statistiques descriptives
df.describe()  

# Requêtes filtrées avec des conditions
df.query('colonne > 10')

# Supprimer des lignes ou des colonnes
df.drop('Nom de colone', axis=1, inplace=True)  # Supprimer une colonne

df.drop(0, axis=0, inplace=True)  # Supprimer la ligne avec l'index 0

# Renommer les colonnes
df.rename(columns={'ancien_nom': 'nouveau_nom'}, inplace=True)

# Remplir les valeurs manquantes (NaN)
df.fillna(value=None, axis=None, inplace=True) 

# Supprimer les lignes contenant des valeurs NaN
df.dropna(axis=0, inplace=True)

# Supprimer les colonnes contenant des valeurs NaN
df.dropna(axis=1, inplace=True)

# Afficher la forme du DataFrame
print(df.shape)

# Supprimer les doublons
df.drop_duplicates(subset=['email'], keep='first', inplace=True)

# Supprimer les lignes contenant NaN dans la colonne 'name'
students.dropna(subset=['name'], axis=0, inplace=True)
