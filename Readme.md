Il semble que les formules dans votre fichier ne s'affichent pas correctement en raison de problèmes de formatage. Voici une version corrigée du contenu avec les formules correctement formatées en LaTeX :

```markdown
# 1. Matrice de confusion

|                | Prédire Positif | Prédire Négatif |
|----------------|----------------|----------------|
| **Classe positive** | TP            | FN            |
| **Classe négative** | FP            | TN            |

# 2. Principales métriques

- **Taux de vrais positifs (Recall)** : \( \text{TPR} = \frac{TP}{P} \)  
- **Taux de faux positifs** : \( \text{FPR} = \frac{FP}{N} \)  
- **Taux de vrais négatifs (Spécificité)** : \( \text{TNR} = \frac{TN}{N} \)  
- **Taux de faux négatifs** : \( \text{FNR} = \frac{FN}{P} \)  

# 3. Taux d'erreur pondéré

\( \pi_P \times \text{FNR} + \pi_N \times \text{FPR} \)  

# 4. Coût de classification

\( C = C_{FP} \times FP + C_{FN} \times FN \)
```

### Explications des corrections :
1. **Formules en LaTeX** : Les formules ont été corrigées pour utiliser une syntaxe LaTeX appropriée. Par exemple, `\text{TPR} = \frac{TP}{P}` est utilisé pour afficher correctement le taux de vrais positifs.
2. **Symboles mathématiques** : Les symboles comme `\times` pour la multiplication et `\frac{}{}` pour les fractions ont été utilisés pour une meilleure lisibilité.
3. **Clarté** : Le texte a été légèrement reformaté pour améliorer la clarté et la cohérence.

Si vous utilisez un environnement Markdown qui supporte LaTeX (comme Jupyter Notebook, certains éditeurs Markdown ou des plateformes comme GitHub), ces formules devraient s'afficher correctement. Si vous avez encore des problèmes, assurez-vous que votre environnement supporte bien le rendu LaTeX.