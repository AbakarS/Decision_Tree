# -*- coding: utf-8 -*-
"""Decision_Tree.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LthZ59m7vfkY4RJaZplXxyZDRkA_fzig

https://www.kaggle.com/code/meghagoriya/iris-decision-tree-finetune

**Importation des librairies**
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
# %matplotlib inline

"""**Importer les données**"""

iris_df = pd.read_csv('https://github.com/GPShiva/Sparks-Internship/blob/main/Iris.csv?raw=true',index_col = 'Id')
iris_df.head()

"""**Separer la variable 'target' des variables explicatives**"""

X = iris_df.drop('Species',axis=1)
y = iris_df.Species
X.shape, y.shape

"""**Spliter les données 25% pour le set de test et 75% pour le set de training**"""

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=100)

"""**Importer les librairies pour l'Arbre des décision et les metrics de performance**"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

"""**Instancier le modéle**"""

classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

"""**Calcul des predictions et comparaison des predictions au valeurs réelles**"""

y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)
print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""On constate une haute précision

**Les parametres utilisés dans la dernière implementation de Decision Tree**
"""

classifier.get_params()

"""### **Changer les parametres pour voir s'il y a un changement dans les scores de précision**

**- ccp_alpha : non-negative float, default=0.0**

Paramètre de complexité utilisé pour l'élagage coût-complexité minimal. Le sous-arbre avec la plus grande complexité de coût inférieure à **ccp_alpha** sera choisi. Par défaut, aucun élagage n'est effectué. Voir [Élagage de complexité de coût minimal](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning) pour plus de détails.
"""

classifier = DecisionTreeClassifier(ccp_alpha=0.01)
classifier.fit(X_train,y_train)

y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Une moindre dimunition du score de précision pour le set de training

**- critère : {“gini”, “entropy”, “log_loss”}, default=”gini”** 

La fonction pour mesurer la qualité d'un split. Les critères pris en charge sont **gini** pour l'impureté Gini et **log_loss** et **entropy** pour le gain d'informations de Shannon
"""

classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,y_train)



y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il n'y a pas de changement dans les scores de précision

**- splitter : {"best", "random"}, default="best"** 

La stratégie utilisée pour choisir la répartition à chaque nœud. Les stratégies prises en charge sont « meilleures » pour choisir la meilleure répartition et « aléatoires » pour choisir la meilleure répartition aléatoire.
"""

classifier = DecisionTreeClassifier(splitter='random')
classifier.fit(X_train,y_train)



y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il n'y a pas de changement dans les scores de précision"""

classifier = DecisionTreeClassifier(splitter='best')
classifier.fit(X_train,y_train)



y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""**- max_depth : int, default=None** 

La profondeur maximale de l'arbre. Si aucun, les nœuds sont développés jusqu'à ce que toutes les feuilles soient pures ou jusqu'à ce que toutes les feuilles contiennent moins de min_samples_split échantillons.
"""

classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train,y_train)



y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Le score de training diminue

**- min_samples_split : int ou float, default=2** 

Le nombre minimum d'échantillons requis pour diviser un nœud interne : Si int, alors considérez min_samples_split comme le nombre minimum. Si float, alors min_samples_split est une fraction et ceil(min_samples_split * n_samples) est le nombre minimum d'échantillons pour chaque fractionnement.
"""

classifier = DecisionTreeClassifier(min_samples_leaf=1)
classifier.fit(X_train,y_train)



y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il n'y a paq de changement dans les résultats de scores de précision

**- min_samples_leaf : int ou float, default=1** 

Le nombre minimum d'échantillons requis pour être à un nœud feuille. Un point de partage à n'importe quelle profondeur ne sera pris en compte que s'il laisse au moins min_samples_leaf échantillons d'apprentissage dans chacune des branches gauche et droite. Cela peut avoir pour effet de lisser le modèle, notamment en régression. Si int, alors considérez min_samples_leaf comme le nombre minimum. Si float, alors min_samples_leaf est une fraction et ceil(min_samples_leaf * n_samples) est le nombre minimum d'échantillons pour chaque nœud.
"""

classifier = DecisionTreeClassifier(min_samples_leaf=1)
classifier.fit(X_train,y_train)

y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il n'y a pas de changement dans les scores de précision

**- min_weight_fraction_leaf : float, default=0.0** 

La fraction pondérée minimale de la somme totale des poids (de tous les échantillons d'entrée) devant être à un nœud feuille. Les échantillons ont le même poids lorsque sample_weight n'est pas fourni.
"""

classifier = DecisionTreeClassifier(min_weight_fraction_leaf=0)
classifier.fit(X_train,y_train)

y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il n'y a pas de changement dans les scores de précision

**- max_features : int, float or {"auto", "sqrt", "log2"}, default=None** 

Le nombre de fonctionnalités à prendre en compte lors de la recherche de la meilleure répartition :

Si **int**, alors considérez les fonctionnalités max_features à chaque fractionnement.

Si **float**, alors max_features est une fraction et int(max_features * n_features) les fonctionnalités sont prises en compte à chaque fractionnement.

Si **"auto"**, alors max_features=sqrt(n_features).

Si **"sqrt"**, alors max_features=sqrt(n_features).

Si **"log2"**, alors max_features=log2(n_features).

Si **None**, alors max_features=n_features.
"""

classifier = DecisionTreeClassifier(max_features='auto')
classifier.fit(X_train,y_train)

y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il y a une augmentation de précision pour le set de test. C'est un résultat interessant"""

classifier = DecisionTreeClassifier(max_features='sqrt')
classifier.fit(X_train,y_train)

y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il n'y a pas de changement dans les scores de précision"""

classifier = DecisionTreeClassifier(max_features='log2')
classifier.fit(X_train,y_train)

y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il n'y a pas de changement dans les scores de précision

**- max_leaf_nodes : int, default=None** 

Développez un arbre avec max_leaf_nodes de la meilleure manière en premier. Les meilleurs nœuds sont définis comme une réduction relative des impuretés. Si aucun, alors nombre illimité de nœuds feuilles
"""

classifier = DecisionTreeClassifier(max_leaf_nodes=10)
classifier.fit(X_train,y_train)

y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il n'y a pas de changement dans les scores de précision

**- min_impurity_decrease : float, default=0.0** 

Un nœud sera dédoublé si ce dédoublement induit une diminution de l'impureté supérieure ou égale à cette valeur.
"""

classifier = DecisionTreeClassifier(min_impurity_decrease=0)
classifier.fit(X_train,y_train)

y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il n'y a pas de changement dans les scores de précision

**- class_weight : dict, liste de dict ou "balanced", default=None** 

Pondérations associées aux classes sous la forme {class_label: weight}. Si aucune, toutes les classes sont censées avoir un poids. Pour les problèmes à sorties multiples, une liste de dicts peut être fournie dans le même ordre que les colonnes de y.
"""

classifier = DecisionTreeClassifier(class_weight='balanced')
classifier.fit(X_train,y_train)

y_train_pred=classifier.predict(X_train)
y_test_pred=classifier.predict(X_test)

print(accuracy_score(y_train,y_train_pred),round(accuracy_score(y_test,y_test_pred),2))

"""Il n'y a pas de changement dans les scores de précision"""