# PANDAS EN PYTHON

## Exemple de Code

```python
import pandas as pd  # Importation de la bibliothèque pandas

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

# Read CSV Files

df = pd.read_csv('./student/student-mat.csv')

# Viewing data

df.head() // Afficher les premières lignes du DataFrame (par défaut, 5 lignes).

df.tail(n) // afiche max n lign 

df.tail() : Afficher les dernières lignes du DataFrame.

df.info() : Afficher des informations sur le DataFrame (types, nombre de valeurs non-nulles, etc.).

df.describe() :La fonction describe() de pandas est utilisée pour obtenir un résumé statistique des colonnes numériques d un DataFrame. Elle fournit des informations statistiques comme la moyenne, l écarttype, les quantiles, etc.

df.query('colonne > 10') : Effectuer des requêtes filtrées avec des conditions.

df.drop() :  en pandas est utilisée pour supprimer des lignes ou des colonnes

df.drop('Nom de colone', axis=1, inplace=True) // ca veut dire suprimier une colone 

df.drop(0, axis=0, inplace=True) // supposons que vous souhaitiez supprimer la ligne avec l index 0

df.rename(columns={'ancien_nom': 'nouveau_nom'}, inplace=True): Renommer des colonnes ou des indices.

df.fillna() : permet de remplir les valeurs manquantes (NaN)

df.fillna(value=None, axis=None, inplace=True) 
// valeur pour remplacer nan
// axis = =0 Remplir les NaN sur les lignes  ou axis =1 Remplir les NaN sur les coloumnes 

df.dropna(axis=0, inplace=True): Supprimer les lignes avec NaN

df.dropna(axis=1, inplace=True): Supprimer les coloumnes avec NaN









