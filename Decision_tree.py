import numpy as np

from sklearn import neighbors, datasets, model_selection, metrics, tree
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

"""
Class to represent a node. Each node has a matrix X of data points and a list y of labels. 
Each node can also have a left child, a right child, a split_index, split_average and a 
most common label.
"""
class Node:
    def __init__(
        self,
        X,
        y,
        left_child = None,
        right_child = None,
        split_index = None,
        split_average = None,
        mcl = None
    ):
        self.X = X
        self.y = y
        self.left_child = left_child
        self.right_child = right_child
        self.split_index = split_index
        self.split_average = split_average
        self.mcl = mcl
    
    #Function to check if the given node is a leaf node.
    def is_leaf(self):
        return self.left_child == None and self.right_child == None

"""
Class to represent a decision tree classifier. Each tree has a root node. 
"""
class Tree:
    def __init__(
            self,
            root = None
        ):
            self.root = root

    """
    Funtion to calculate gini value of a node, based on the node's y list of labels. 
    """
    def gini(self, y):
        
        #get the different values of y (h and g), and how many occurances these have
        values, counts = np.unique(y, return_counts=True)
        gini = 1
        
        for i in range(len(values)):
            prob = counts[i]/len(y)
            gini -= pow(prob,2)
        return gini


    """    
    Funtion to calculate entropy value of a node, based on the node's y list of labels. 
    """
    def entropy(self, y):

        #get the different values of y (h and g), and how many occurances these have
        counts = np.unique(y, return_counts=True)
        entropy = 0

        for i in range(len(counts[0])):
            prob = counts[1][i]/len(y)
            entropy += prob*np.log2(prob)
        return (-entropy)

    
    """
    Function to calculate information gain from a split, based on the given impurity measure.
    """
    def information_gain(self, parent_y, left_y, right_y, impurity_measure):
        
        weight_left = len(left_y)/len(parent_y)
        weight_right = len(right_y)/len(parent_y)
        
        information = 0
        impurity_parent = 0
        
        #calculate the information based on the impurity measure
        if impurity_measure == "gini":
            gini_left = self.gini(left_y)
            gini_right = self.gini(right_y)
            impurity_parent = self.gini(parent_y)
            information = weight_left*gini_left + weight_right*gini_right   
        else:
            en_left = self.entropy(left_y)
            en_right = self.entropy(right_y)
            impurity_parent = self.entropy(parent_y)
            information = weight_left*en_left + weight_right*en_right
        
        gain = impurity_parent - information
        return gain


    """
    Function to check if a node has identical features.
    """
    def has_identical_features(self, X):
        
        for i in range(len(X[0])):
            #make a list of all values in the current column
            features = [col[i] for col in X]
            
            #if all the values in this column are equal, return True
            if np.all(features == features[0]):
                return True, features
            
        return False


    """
    Funtion to find a node's most common label.
    """
    def most_common_label(self, node):
        
        #if the given node is a leaf, all y-values are the same
        if node.is_leaf():
            return node.y[0]
    
        else:
            values, counts = np.unique(node.y, return_counts = True)
            
            if counts[0] > counts[1]:
                return values[0]
            else:
                return values[1]

            
    """
    Function to divide given data based on the split index and average at each node.
    """
    def divide_data(self, node, X, y):
        
        #make a list of all values in the split-index-column
        features = [col[node.split_index] for col in X]
        
        left_y = [] 
        left_X = []
        right_y = []
        right_X =[]
        
        #loop through all the values in the column
        for i in range(len(features)):
            
            #split the data based on the split average
            if features[i] < node.split_average:
                left_X.append(X[i])
                left_y.append(y[i])
                
            else:
                right_X.append(X[i])
                right_y.append(y[i])
                
        return left_X, left_y, right_X, right_y


    """
    Function to prune a tree, starting from the root of the tree, based on given prune data.
    """
    def prune(self, tree_root, prune_X, prune_y):

        #if the root of the current split is a leaf, return the accuracy of this node
        if tree_root.is_leaf():
            return prune_y.count(tree_root.mcl) #self.calc_error(tree_root.mcl, prune_y)
        
        #use divide_data to divide the prune data
        left_X, left_y, right_X, right_y = self.divide_data(tree_root, prune_X, prune_y)
        
        #get the accuracy of the prune data in the child nodes with recursion
        left_acc = self.prune(tree_root.left_child, left_X, left_y)
        right_acc = self.prune(tree_root.right_child, right_X, right_y)
        
        parent_acc = prune_y.count(tree_root.mcl) #self.calc_error(tree_root.mcl, prune_y)
        
        #if none of the prune data went to one of the child nodes, pruning won't occur
        if left_acc != None and right_acc != None:
            
            #if parent accuracy is better than the child nodes', prune this sub tree.
            if parent_acc > left_acc + right_acc:
                
                tree_root.left_child = None
                tree_root.right_child = None
                return parent_acc
            
        return left_acc + right_acc


    """
    Function to split the given node into two child nodes, left and right, 
    based on the impurity measure
    """
    def split(self, node, impurity_measure):

        #get the number of feature rows and columns
        num_f_rows, num_f_cols = np.shape(node.X)
        
        opt_gain = 0
        
        for i in range(num_f_cols):
            
            #make a list of all values in the current column
            features = [col[i] for col in node.X]
            
            #calculate the value to do the splitting after
            feature_average = np.average(features)
                
            left = Node([],[])
            right = Node([],[])

            #loop through each value in the column, and assign them to a child node, based on the feature average
            for j in range(num_f_rows):
                if features[j] < feature_average:
                    left.X.append(node.X[j])
                    left.y.append(node.y[j])
                    
                else:
                    right.X.append(node.X[j])
                    right.y.append(node.y[j])
                    
            #information gain for the current split       
            gain = self.information_gain(node.y, left.y, right.y, impurity_measure)
             
            #check if current split is the optimal one    
            if (gain > opt_gain):
                opt_gain = gain
                node.left_child = left
                node.right_child = right
                node.split_index = i
                node.split_average = feature_average
            
        return node.left_child, node.right_child 


    """
    Function to grow the tree from a given root node
    """
    def grow_tree(self, node, impurity_measure):
        
        #assign the current node it's most common label
        node.mcl = self.most_common_label(node)
        
        #if the entropy of a node is 0, it's y values are pure
        if self.entropy(node.y) == 0:
            node.mcl = node.y[0]
            return 

        elif self.has_identical_features(node.X):
            node.mcl = most_common_label(node)
            return 

        else: 
            #split the node
            left, right = self.split(node, impurity_measure)
            
            #continue growing the tree
            self.grow_tree(left, impurity_measure)
            self.grow_tree(right, impurity_measure)


    """
    Function to create a decision tree based on given X and y data, impurity_measure and pruning
    """
    def learn(self, X, y, impurity_measure="entropy", pruning=False):

        seed = 300
        
        #splitting the training data into training and pruining
        X_train, X_prune, y_train, y_prune = model_selection.train_test_split(
        X, y, test_size = 0.3, random_state = seed)
        
        #creating the root of the tree
        self.root = Node(X_train, y_train)
        
        #grow the tree
        self.grow_tree(self.root, impurity_measure)
        
        #prune tree if pruning is true
        if pruning:
            self.prune(self.root, X_prune, y_prune)
        
        return self


    """
    Function to calculate total accuracy of the tree, based on given data
    """
    def calc_total_accuracy(self, X, y):
        pred = []
        
        for x in X:
            pred.append(self.predict(x, self.root))
         
        return metrics.accuracy_score(y, pred)
            
    
    """
    Function to predict the label of a given x data point
    """
    def predict(self, x, node):

        if node.is_leaf():
            return node.mcl
        
        if (x[node.split_index] < node.split_average):
            return self.predict(x, node.left_child)
        else:
            return self.predict(x, node.right_child)


"""
Function to select the best model, based on the accuracies given
"""
def select_best_model(val_accuracies):
    max_model = max(val_accuracies, key=val_accuracies.get)
    max_acc = val_accuracies.get(max_model)
    return max_model

data = pd.read_csv("magic04.data", header=None)

X = data.values[:,:-1].tolist()
y = data.values[:,-1].tolist()

seed = 300

#splitting the dataset into training and test/validation
X_train, X_val_test, y_train, y_val_test = model_selection.train_test_split(
X, y, test_size = 0.3, random_state=seed)
        
#splitting the validation/testing data 
X_val, X_test, y_val, y_test = model_selection.train_test_split(
X_val_test, y_val_test, test_size = 0.5, random_state = seed)


print("Welcome to my decision tree! :)\n_______________________________\n\nFirst, we need to grow the tree using our training dataset:")
print("Growing tree ...\n")

tree_en = Tree().learn(X_train, y_train)
tree_gi = Tree().learn(X_train, y_train, "gini")

print("Tree is fully grown!\n")

print("Next, we should prune the tree a little bit. It is way too big!\nPruning tree ...\n")

tree_en_pr = Tree().learn(X_train, y_train, "entropy", True)
tree_gi_pr = Tree().learn(X_train, y_train, "gini", True)

print("Tree is pruned!\n_______________________________\n")
print("Now, we want to check if our tree is any good for a different set of data. We will call this out validation dataset.\n")
print("Accuracy on validation dataset with ...")

val_acc_en = tree_en.calc_total_accuracy(X_val, y_val)
val_acc_gi = tree_gi.calc_total_accuracy(X_val, y_val)
val_acc_en_pr = tree_en_pr.calc_total_accuracy(X_val, y_val)
val_acc_gi_pr = tree_gi_pr.calc_total_accuracy(X_val, y_val)

print(f"- entropy and without pruning: {val_acc_en*100}%")
print(f"- gini and without pruning: {val_acc_gi*100}%")#train_acc_gi_val
print(f"- entropy and with pruning: {val_acc_en_pr*100}%")#train_acc_en_pr_val
print(f"- gini and with pruning: {val_acc_gi_pr*100}%")#train_acc_gi_pr_val

print("\nThe accuracy with the validation dataset is slighly different when using the different settings, so let's choose the best one.\n")

val_accuracies = {
    "Entropy without pruning" : train_acc_en_val,
    "Gini without pruning" : train_acc_gi_val,
    "Entropy with pruning" : train_acc_en_pr_val,
    "Gini with pruning" : train_acc_gi_pr_val
}

best_model = select_best_model(val_accuracies)

print(f"{best_model} was the best setting!\n")
print("Now, lets test our implementation with the test dataset on our chosen model.\n")

test_acc = tree_gi_pr.calc_total_accuracy(X_test, y_test)

print(f"Accuracy on test dataset with the best model: {test_acc*100}%\n")
print("Pretty good! :)")
print("_______________________________\n")

print("Growing a decision tree with sklearn's Decision Tree Classifier ...")

#tree with gini
clf_gini = tree.DecisionTreeClassifier()
clf_gini = clf_gini.fit(X_train, y_train)

#tree with entropy
clf_en = tree.DecisionTreeClassifier()
clf_en = clf_en.fit(X_train, y_train)

print("Done! That was fast!")

y_pred_gini = clf_gini.predict(X_val)
y_pred_en = clf_en.predict(X_val)

score_gini = metrics.accuracy_score(y_val, y_pred_gini)
score_en = metrics.accuracy_score(y_val, y_pred_en)
print(f"Accuracy score with gini: {score_gini*100}%")
print(f"Accuracy score with entropy: {score_en*100}%")
