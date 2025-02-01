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




