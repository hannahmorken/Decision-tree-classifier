# Decision-tree-classifier

## INF264

### Project 1: Implementing decision trees

Write a computer program that solves the following tasks. Then write a short report that explain your approach and design choices (Tasks 1.1-1.3) and the results of your experiments as well as the experimental procedure you used (Tasks 1.4-1.5).

#### 1.1 Implement a decision tree learning algorithm from scratch
Implement a greedy algorithm for learning decision trees: 

• If all data points have the same label
– return a leaf with that label

• Else if all data points have identical feature values
– return a leaf with the most common label 

• Else
– choose a feature that maximizes the information gain
– split the data based on the value of the feature and add a branch
for each subset of data
– for each branch
∗ call the algorithm recursively for the data points belonging to the particular branch
You should use entropy as the impurity measure. Your implementation should have two functions that the users can use:
1. learn(X, y, impurity_measure=’entropy’) 
2. predict(x, tree)

The function learn learns a decision tree classifier from a data matrix X and a label vector y. We consider the classification task so you can assume that y consists of categorical variables. You can assume that X consists of continuous features.
The function predict predicts the class label of some new data point x.
Note: If you implement your tree in object-oriented fashion, it is not necessary to include the argument tree.
Note: For debugging and visualisation purposes, it may be useful to implement a function that prints the tree.


#### 1.2 Add Gini index
Implement Gini index as an alternative impurity measure. To use the Gini index, you should call your learning function like learn(X, y, impurity_measure=’gini’).


#### 1.3 Add reduced-error pruning
Decision tree learning is prone to overfitting. To fix this, extend your algorithm to tackle overfitting using reduced-error pruning:
• Divide data to training and pruning data
1. Use the training data to build a full decision tree T∗ (using the algorithm from Section 1.1) 
2. For each subtree T of T∗
• If replacing the subtree with the majority label in T (based on training data) does not decrease accuracy on the pruning data
– Replace T with a leaf node that predicts the majority class
You will need to extend your function’s argument list with an additional parameter called prune which should by default be set to False.
Since pruning should be done inside the learn method, the pruning set is a subset of the training set. You can add an additional parameter that specifies which part of the data should be used as a pruning set.
Note that reduced-error pruning starts from the leaves and proceeds bottom-up.

#### 1.5 Compare to an existing implementation
Compare your implementation to some existing decision tree implementation. How does your implementation fare against this implementation in terms of accuracy and speed? Can you explain the (possible) differences?
Note: You can compare to, for example, DecisionTreeClassifier from sklearn.
