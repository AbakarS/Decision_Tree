### Decision_Tree

#### L’arbre de decision

Un arbre de décision est une méthode d’apprentissage automatique supervisée qu’on peut utiliser à la fois pour les problèmes de régression et de classification. Il s’agit d’un modèle d’apprentissage arborescent où les nœuds internes (les nœuds internes sont aussi appelés des nœuds de décision) représentent les caractéristiques d’un ensemble de données, les branches représentent l’ensemble de règles de décision et chaque nœud feuille (ce nœud feuille est aussi appelé nœud terminal) contient la classe à prédire ou la variable cible.

L’arbre de décision est alors composé  : 

- D’un nœud racine, c’est le nœud à partir duquel on construit l’arbre : C’est par là qu’on fasse entrer les données. 
- Des nœuds internes qui conditionnent la suite de l’arbre en se basant sur des règles de décision. Les nœuds qui débouchent sur d’autres nœuds sont appelés des nœuds parents tandis que les nœuds découlant d’autres nœuds sont appelés des nœuds enfants et ainsi de suite. Un nœud qui n’a aucun nœud enfant est appelé nœud feuille prédisant la sortie;
- Des branches qui connectent les nœuds entre eux. 
- 
Prenons l’exemple suivant pour comprendre les règles de décision d’un arbre de décision : Cet exemple est présenté tel qu’il est récupéré de l’article publié en 2019 par Professeure  Neila Mezghani 

Figure

![image](https://user-images.githubusercontent.com/88625171/182031990-74a6bc08-778b-42f0-9c3c-5ced75b9c70b.png)


L’arbre indique si une banque peut accorder ou non à un demandeur un prêt en se basant sur des caractéristiques telles que le salaire moyen (MoySal), l’âge et la possession d’autres comptes. 
En partant de cet exemple nous présentons le processus de construction de l’Arbre de Décision dans la figure (Figure 1) ci-dessous : 

Figure : Exemple Arbre de Décision (Decision Tree)
 
Source : Neila Mezghani, TELUQ, 2019
Suite à cet Arbre de Décision, on peut citer les règles de décision suivantes : 

- Si MoySal > 50000$ alors prêt = Oui
- Si MoySal < 50000$ et âge < 25 alors prêt = Non
- Si MoySal <50000$ et âge > 25 et Autre comptes = Oui, alors prêt = Oui
- Si MoySal < 50000$ et âge > 25 et Autres comptes = Non, alors prêt = Non
	
Pour construire un arbre de décision plusieurs questions se posent, parmi lesquelles :
	
- Quelle variable passe en premier et discrimine mieux l’ensemble de données ?
- Vue les situations décrites ci-haut, quelle structure d’arbre faudra-t-il utilisée ?
- Pour maximiser la robustesse des arbres, quel critère utilise-t-on ?
	
Pour répondre à certaines de ces questions, on utilise l’entropie croisée et le gain d’information ou Indice Génie. Pour maximiser la robustesse des arbres de decision, on utilise la technique d’élagage . 

Les avantages que procurent les arbres de décision sont multiples :
- Les arbres decision sont faciles à visualiser et du coup facilement interprétable. 
- Les arbres de décision ne nécessitent pas la normalisation des caractéristiques en amont. 
- Ce classificateur gère en plus parfaitement les données qualitatives que quantitatives. 
	
Les arbres de décision ont aussi des inconvénients parmi lesquels on cite :

- Les arbres de décision sont peu robustes, un petit changement dans le set de training pourrait entrainer un grand changement dans la structure de l’arbre et par conséquent le résultat final;
	
- L’arbre idéal est l’arbre dont la profondeur est le plus petit possible, c’est-à-dire avec moins de divisions, car plus il y a des divisions plus l’arbre se complexifie, et plus il y a un risque d’Overfiting . Cependant, il y a des techniques tels que l'élagage, la définition du nombre minimum d'échantillons requis à un nœud feuille ou la définition de la profondeur maximale de l'arbre sont nécessaires pour éviter ce problème.
-  
Pour pallier à tous ces inconvénients d’arbres de décision, les forêts aléatoires (ou Random Forest) ont vu le jour. 
Comment coder un algorithme d’Arbre de Décision pour la classification

Pour construire un Arbre de Décision pour un problème de classification, il faut tout simplement utiliser la librairie adéquate Sklearn en Python. Affichons ci-dessous la fonction Scikit Learn dédiée à la construction de l’Arbre de Décision et expliquons l’importance de chacun de ses différents paramètres :

**sklearn.tree.DecisionTreeClassifier(*,criterion='gini',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features=None,random_state=None,max_leaf_nodes=None,min_impurity_decrease=0.0,class_weight=None,ccp_alpha=0.0)**

- **Criterion : gini, entropy, log_loss** : la fonction de coût, c’est le critère de performance pour construire l’Arbre de Décision. On peut choisir parmi les trois fonctions (gini, entropy et log_loss). Il faut savoir que par défaut Sklearn utilise « gini » comme criterion. Donc, on va expliquer brièvement c’est que c’est gini ? 

**Gini** : Il s’agit d’un critère permettant de mesurer la qualité ou la pureté d’une division. En effet, le criterion de gini permet de calculer l’impureté de chaque nœud d’un Arbre de Décision. A partir de cet indice d’impureté, on calcule le coût d’un nœud, ce qui permettra de construire l’Arbre, en commençant par le nœud a le plus faible coût. Considérez un ensemble de données F qui contient des échantillons de k classes. La probabilité d'appartenance d'échantillons à une classe j à un nœud donné peut être notée pi. Alors l'impureté de Gini est définie comme :

![image](https://user-images.githubusercontent.com/88625171/182032363-49c0f2eb-90e3-4520-8b9b-b4b028c6155b.png)

NB : l’impureté de gini est comprise entre 0 et 0.5
Le coût de nœud : c’est le coût de la règle de décision derrière la construction de ce nœud, il nous permet de savoir à quel point cette décision est bonne ou pas. Cliquer ici pour plus détails sur l’indice de gini et le coût d’un nœud. Consulter ce lien pour l’entropy. 

**Splitter : best, random** : ce paramètre nous permet de choisir comment on va séparer notre data set et comment on va construire notre Arbre. Il s’agit de la stratégie permettant d’effectuer la répartition à chaque nœud de l’arbre. Par défaut le splitter est positionné sur « best », ce qui veut dire que l’algorithme va choisir la meilleure caractéristique à séparer pour construire notre Arbre. Si en revanche, on choisit « random », l’algorithme va choisir une variable au hasard à séparer. Il vaut mieux garder le splitter en « best » pour une meilleure séparation. Mais on peut utiliser « random » au cas où on voudrait économiser le temps de calcul. 

**max_depthint :** C’est l’un des paramètres le plus important de notre Arbre, il s’agit de la profondeur de notre Arbre de Décision. Ce paramètre va jouer sur le nombre de règles de décision qu’on devrait avoir et sur la complexité de notre Arbre. Si on ne choisit pas de profondeur à notre Arbre, l’algorithme va continuer à créer de séparation jusqu’à ce qu’il ne trouve plus de séparation possible. Il faut savoir que plus l’Arbre est profond, plus il est performant mais moins généralisable sur le set de test. Bref, plus l’Arbre est profond plus on tend vers un surapprentissage, et plus l’Arbre est moins profond on tend alors vers un sous-apprentissage. Donc, il faut varier ce paramètre en faisant plusieurs entrainements et garder le meilleur modèle généralisable.  Par défaut ce paramètre est réglé sur None.

**min_samples_split** : Ce paramètre nous permettra de choisir le nombre minimal d’échantillon que le modèle doit avoir pour pouvoir faire une nouvelle séparation. Si le nœud ne trouve pas ce nombre minimal d’échantillon, il créera une feuille. On a deux façons de choisir ce paramètre en mettant un entier ou un flottant (un nombre à virgule entre 0 et 1). Alors, si on choisit une valeur faible, on aura un modèle plus complexe, et si on choisit une haute valeur on obtient un modèle plus simple. Par défaut, il est fixé à 2. 

**min_samples_leaf** : Ce paramètre va déterminer le nombre minimal d’échantillons requis pour créer un nœud feuille. Donc, lors d’une séparation, si l’algorithme voit que la population dans une branche ne pourra pas créer un nœud feuille, il va simplement ne pas effectuer cette séparation. Par ailleurs, plus la valeur de min_samples_leaf sera petite plus on aura un modèle complexe mais performant (cela pourrait tendre le modèle vers un surapprentissage ou Overfitting), parce qu’il faudra moins d’échantillons pour créer un nœud feuille, plus la valeur est grande plus, on créera des modèles simples (cela va tendre le modèle vers un sous apprentissage ou Underfitting). Par défaut ce paramètre est fixé à 1 par Sklearn. 
	
**min_weight_fraction_leaf** : Par défaut tous les échantillons du set de training ont les mêmes poids d’apprentissage, mais il peut arriver qu’on veille donner des poids différents à chaque observation de notre set d’entrainement, parce qu’il y a des échantillons qui doivent être résolu à tous prix et d’autres qui sont moins importants . Donc, on accorde aux différentes observations de poids d’importance. Dans ce cas, un entrainement donne à l’algorithme un vecteur de poids d’apprentissage. On peut alors définir un poids minimum pour créer un nœud feuille. Plus ce paramètre est petit plus le modèle sera performant mais moins généralisable, plus il sera grand plus le modèle sera simple mais moins performant. 
	
**max_features** : ce paramètre contrôle le nombre de features à tester afin de trouver la meilleure séparation. Il est fixé par défaut à « None ». Si ce paramètre prend un « entier », cela équivaut au nombre de test à réaliser pour trouver la meilleure séparation. En revanche, si ce paramètre est fixé à « auto », on va tester toutes les variables données en entrée. Par défaut, ce paramètre est fixé à None, il est conseillé de laisser ce paramètre à « None » ou « auto » afin de pouvoir tester toutes les variables pour trouver la meilleure séparation. 
	
**random_state**: il contrôle le caractère aléatoire du classificateur, par défaut il est fixé à « None ». Il faut toujours fixer ce paramètre sur un entier pour avoir les mêmes résultats à chaque simulation du classificateur. 

**max_leaf_nodes** : ce paramètre nous permet de choisir le nombre maximum de feuilles de notre arbre. Il est important pour contrôler la complexité de l’arbre. Plus ce paramètre est grand, plus l’arbre sera complexe, plus il est petit, plus l’arbre sera simple. Par défaut, il est fixé à « None ». 
	
**min_impurity_decrease** : ce paramètre permet de contrôler si l’ajout d’une règle ajoutera assez de pureté afin de la conserver. Ce que fait min_impurity_decrease est d'arrêter un fractionnement si le montant de la diminution du fractionnement est inférieur au montant saisi. Superbe article sur ce paramètre. 
	
**class_weight** : ce paramètre permet de donner un poids diffèrent aux classes de notre set de training. Ce paramètre est important lorsque nous avons un ensemble de données dont les classes sont déséquilibrées. Si ce paramètre est fixé à « None », toutes les classes vont avoir un poids. Si par exemple, nous avons des classes déséquilibrées, ce paramètre nous permettra d’avoir un poids différent pour chaque classe qu’on essaye de modéliser. Il faut savoir que l’algorithme favorisera l’entrainement dans la classe où il y a le plus de poids. Donc, pour contrebalancer le fait que nous avons un ensemble de données déséquilibrées, nous allons mettre un poids diffèrent à chaque classe pour faire en sorte que la représentation de notre erreur dans notre data set lors de l’entrainement soit la même pour chaque classe. 
	 
**ccp_alpha** : ce paramètre permet de supprimer des parties de l’Arbre qui ajoute de la complexité au modèle sans apporter beaucoup de valeur supplémentaire. Le but de cette suppression c’est de rendre ce classificateur plus simple et plus généralisable en conservant une performance acceptable. Par défaut, il est fixé à zéro. 
