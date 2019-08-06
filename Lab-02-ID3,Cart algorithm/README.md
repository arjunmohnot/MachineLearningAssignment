ID3
===

- Decision tree algorithms transfom raw data to rule based decision making trees. Herein, ID3 is one of the most common decision tree algorithm. Firstly, It was introduced in 1986 and it is acronym of Iterative Dichotomiser.

- First of all, dichotomisation means dividing into two completely opposite things. That’s why, the algorithm iteratively divides attributes into two groups which are the most dominant attribute and others to construct a tree. Then, it calculates the entropy and information gains of each atrribute. In this way, the most dominant attribute can be founded. After then, the most dominant one is put on the tree as decision node. Thereafter, entropy and gain scores would be calculated again among the other attributes. Thus, the next most dominant attribute is found. Finally, this procedure continues until reaching a decision for that branch. That’s why, it is called Iterative Dichotomiser.

* We can summarize the ID3 algorithm as illustrated below: 
  - Entropy(S) = ∑ – p(I) . log2p(I)
  - Gain(S, A) = Entropy(S) – ∑ [ p(S|A) . Entropy(S|A) ]

- ``
So, decision tree algorithms transform the raw data into rule based mechanism. ID3 can use nominal attributes whereas most of common machine learning algorithms cannot. However, it is required to transform numeric attributes to nominal in ID3. Besides, its evolved version C4.5 exists which can handle nominal data. Even though decision tree algorithms are powerful, they have long training time. On the other hand, they tend to fall over-fitting. Besides, they have evolved versions named random forests which tend not to fall over-fitting issue and have shorter training times.
``

![Decision-Trees-modified-1](https://user-images.githubusercontent.com/33459977/62525534-36a49f00-b855-11e9-934f-5c8ed41371d0.png)

Cart
===
- An algorithm can be transparent only if its decisions can be read and understood by people clearly. Even though deep learning is superstar of machine learning nowadays, it is an opaque algorithm and we do not know the reason of decision. Herein, Decision tree algorithms still keep their popularity because they can produce transparent decisions. ID3 uses information gain whereas C4.5 uses gain ratio for splitting. Here, CART is an alternative decision tree building algorithm. It can handle both classification and regression tasks. This algorithm uses a new metric named gini index to create decision points for classification tasks.

- Gini index
  - Gini index is a metric for classification tasks in CART. It stores sum of squared probabilities of each class. We can formulate it as illustrated below.
  - Gini = 1 – Σ (Pi)2 for i=1 to number of classes
 ![cart-step-6](https://user-images.githubusercontent.com/33459977/62525868-c9ddd480-b855-11e9-8e0d-bc087de34192.png)
