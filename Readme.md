
# Data Preprocessing EN PYTHON

## les etape nessecaires

```python

1 ) Importer les Bibliothèques Nécessaires

   import pandas as pd
   import numpy as np
   import seaborn as sns
   import matplotlib.pyplot as plt

2 )  Lire le Jeu de Données

  df = pd.read_csv("./student/train.csv")
  df.head()

3 ) Vérification de la Cohérence des Données (Sanity Check)

 df.isnull().sum() // verifier la somme des value que non manquant
 df.isnull().sum() / df.shape[0] * 100 // pourcentage de value que non manquant

4 ) Analyse Exploratoire des Données (EDA)

- df.describe()
- // Comprendre la distribution des données et  Identifier les valeurs aberrantes(outliers)
   outliers : Ces valeurs sont souvent situées loin de la majorité
    ex :
    for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df, x=i)
    plt.show()

- // scatter plots est d’aider à explorer les relations entre les variables et à découvrir des outliers (valeurs aberrantes).
   for i in ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
           'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
           'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
           'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
           'MiscVal', 'MoSold', 'YrSold', 'SalePrice']:
    sns.scatterplot(data=df, x=i, y='SalePrice')
    plt.show()

  -df.corr() // detetct la correlation entre les variable

5) Traitement des Valeurs Manquantes

  // Remplacement des valeurs manquantes par ex la moyenne
  df.fillna(df.mean(), inplace=True)

  // ou  Supprimer les lignes contenant des valeurs manquantes
  df.dropna(inplace=True)

6 ) Traitement des Valeurs Aberrantes (Outliers)

7 ) Traitement des Doublons et Valeurs Inutiles

8 ) Normalisation des Données

9 ) Encodage des Données Catégorielles

```

## PANDAS EN PYTHON

## Exemple de Code

```python

import pandas as pd  

# Création d'un dataset sous forme de dictionnaire
mydataset = {
    'cars': ["BMW", "Volvo", "Ford"],
    'passings': [3, 7, 2]
}

# Transformation du dataset en DataFrame (est utilisé pour transformer un dataset sous forme de tableau en une structure matricielle.)

myvar = pd.DataFrame(mydataset)
df.columns = ['student_id', 'age'] // on peut faire ca juste pour renomer les colomnes de DataFrame
#refer to the row index:
print(myvar.loc[0])

print(df.loc[[0, 1]]) // afiche 2 lign 

------------------------------------------------------------------------


# Exemple de création d'une Série pandas
#  ca veut dire affichier le tableau avec index 

a = [1, 2, 3, 4, 5]
myvar = pd.Series(a)

// output
0 1
1 2
2 3
....

myvar = pd.Series(a, index = ["x", "y", "z"])  
// dans ce cas on peut donner index 
# Affichage de la Série
print(myvar)

------------------------------------------------------------------------

# Fonction en pandas

- df = pd.read_csv('./student/student-mat.csv')


- df.head() // Afficher les premières lignes du DataFrame (par défaut, 5 lignes).

- df.tail(n) // afiche max n lign 

- df.tail() : Afficher les dernières lignes du DataFrame.

- df.info() : Afficher des informations sur le DataFrame (types, nombre de valeurs non-nulles, etc.).

- df.describe() :La fonction describe() de pandas est utilisée pour obtenir un résumé statistique des colonnes numériques d un DataFrame. Elle fournit des informations statistiques comme la moyenne, l écarttype, les quantiles, etc.

- df.query('colonne > 10') : Effectuer des requêtes filtrées avec des conditions.

- df.drop() :  en pandas est utilisée pour supprimer des lignes ou des colonnes

- df.drop('Nom de colone', axis=1, inplace=True) // ca veut dire suprimier une colone 

- df.drop(0, axis=0, inplace=True) // supposons que vous souhaitiez supprimer la ligne avec l index 0

- df.rename(columns={'ancien_nom': 'nouveau_nom'}, inplace=True): Renommer des colonnes ou des indices.

- df.fillna() : permet de remplir les valeurs manquantes (NaN)

- df.fillna(value=None, axis=None, inplace=True) 
// valeur pour remplacer nan
// axis = =0 Remplir les NaN sur les lignes  ou axis =1 Remplir les NaN sur les coloumnes 

- df.dropna(axis=0, inplace=True): Supprimer les lignes avec NaN

- df.dropna(axis=1, inplace=True): Supprimer les coloumnes avec NaN


- df.shape : donner combien de lign et de columnes

- drop_duplicates(subset=['email'], keep='first') //  pour drop duplicue value

s- tudents.dropna(subset =['name'],axis = 0) // Supprimer les lignes avec NaN juste pour la coloumnes name

- astype() // Vous pouvez utiliser cette fonction pour convertir une colonne en un autre type
// df['colonne'] = df['colonne'].astype(nouveau_type)

- pd.concat([df1, df2], ignore_index=True) // fusionner deux dataFrame

```

# NumPy EN PYTHON

## Exemple de Code NumPy

```python

- arr = np.array([1, 2, 3, 4, 5])  // Create a NumPy array

- arr = np.array([[1, 2, 3], [4, 5, 6]]) // Create a 2-D array

- arr[1:5] // Slice elements from index 1 to index 5-1

- arr[-3:-1] // Slice from the index 3-1  from the end to index 1 from the end

- arr[1:5:2] //  Slice elements from index 1 to index 5-1 avec pas = 2

- arr = np.array([1, 2, 3, 4], dtype='S') // Create an array with data type string

- x = arr.copy() 

- x = arr.view() // if arr change x change 


