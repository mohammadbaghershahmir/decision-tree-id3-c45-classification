import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import  graphviz
file_path = "E:/Ml-Project/HW2_Shahmir_40221570004/bank.csv"
df = pd.read_csv(file_path, sep=';')
print(df.head(5))

# Change datatype from string to float
stringColumns = df.select_dtypes(include=['object']).columns

for columnName, columnValue in df.items():
    if columnName in stringColumns:
        df[columnName] = LabelEncoder().fit_transform(df[columnName])

# Use "duration" as the class column
class_column = df['marital']

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.20)

Test_X = test.drop(['marital'], axis=1)
Test_Y = test[['marital']]

Train_X = train.drop(['marital'], axis=1)
Train_Y = train[['marital']]



####################      ID3  Algorithm ######################################

class DecisionTree:
    """DataStructure for storing out the decision tree."""
    def __init__(self, attrib_name, tabs):
        self.attrib_name = attrib_name
        # Children would be edge_value: Decision Tree
        self.children = {}
        self.class_ = None
        self.tabs = tabs

    def __str__(self):
        return '['+self.attrib_name+'] --> '+DecisionTree.tree_print(self.children, self.tabs)

    __repr__ = __str__

    @staticmethod
    def tree_print(_dict, tabs):
        """Pretty print a tree with tabs and new lines."""
        _tabs = '\n' + ' ' * tabs
        return _tabs + (_tabs.join('{}: {}'.format(k, v) for k, v in _dict.items()) + '}')

class Leaf:
    """Leaf Nodes for the tree that has class set"""
    def __init__(self, class_):
        """ Create an object for Leaf class."""
        self.class_ = class_

    def __str__(self):
        """ String repr for Leaf."""
        return str(self.class_)

    __repr__ = __str__

class MyDecisionTreeClassifier:
    """Classifier class that classifies and predicts data."""
    def __init__(self):
        self.data = None
        self.class_data = None
        self.root = None

    def fit(self, data, class_):
        """Fit the training data to form the decision tree."""
        self.data = pd.DataFrame(data)
        self.class_data = class_
        self.root = self.build_tree(self.data, self.class_data)
        return self

    def predict(self, df_test_data):
        """Predict the class in the test data with the built tree."""
        predict = []
        for _, row in df_test_data.iterrows():
            temp = self.root
            columns = row.axes[0].tolist()
            columns.remove('y')

            while temp.class_ is None:
                for _attrib in columns:
                    attrib_val = row[_attrib]
                    if _attrib == temp.attrib_name:
                        for key in temp.children:
                            if key == attrib_val:
                                temp = temp.children[key]
                                break
                        if temp.class_ is not None:
                            break
            predict.append(temp.class_)
        return predict

    def build_tree(self, rows, class_data):
        """Build a decision tree with the pandas rows."""
        class_ = MyDecisionTreeClassifier.is_all_same_class(rows)
        if rows.shape[0] == 0 or class_ is not None:
            return Leaf(class_)

        _best_attrib = MyDecisionTreeClassifier.best_attrib(rows)
        decision_tree = DecisionTree(_best_attrib, len(_best_attrib) + 5)
        # For all the labels of the best_attrib, we should create
        # children
        for _label in rows[_best_attrib].unique().tolist():
            sub_rows = rows[rows[_best_attrib] == _label]
            sub_rows = sub_rows.drop([_best_attrib], axis=1)
            decision_tree.children[_label] = self.build_tree(sub_rows, class_data)
        return decision_tree

    def print_tree(self):
        """Prints the decision tree with proper indentations."""
        if self.root is None:
            raise Exception("Tree is empty. Please fit the training data to visualize")
        self._print_tree(self.root)
    def _print_tree(self, node, indent=0):
        """Print helper function for printing the decision tree."""
        if isinstance(node, DecisionTree):
            print("  " * indent + f"[{node.attrib_name}]")
            for child_label, child_node in node.children.items():
                print("  " * (indent + 1) + f"{child_label}: ", end="")
                self._print_tree(child_node, indent + 1)
        elif isinstance(node, Leaf):
            print("  " * indent + f"Class: {node.class_}")

    @staticmethod
    def is_all_same_class(rows):
        """Checks whether all the records have the same class."""
        distinct_classes = rows['y'].unique().tolist()
        return distinct_classes[0] if len(distinct_classes) == 1 else None

    @staticmethod
    def best_attrib(rows):
        """This finds the best attribute for tree split."""
        attribs = rows.columns.values.tolist()
        classes = rows['y'].unique().tolist()
        # Removing class
        attribs.remove('y')
        min_entropy = 100  # Some random large value for minimalizing
        min_attrib = None

        for attrib in attribs:
            total_entropy = 0

            for _label in rows[attrib].unique().tolist():
                rows_label = rows[rows[attrib] == _label]
                label_count = rows_label.shape[0]
                label_prob = label_count / rows.shape[0]
                entropy = 0
                for _class in classes:  # Yes or No
                    rows_classes = rows_label[rows_label['y'] == _class]
                    class_count = rows_classes.shape[0]
                    prob = class_count / label_count
                    entropy -= 0 if prob == 0 else prob * np.log2(prob)
                total_entropy += label_prob * entropy
            if total_entropy < min_entropy:
                min_entropy = total_entropy
                min_attrib = attrib
        return min_attrib

###########################  Execution   #################################
# ایجاد یک نمونه از کلاس MyDecisionTreeClassifier
clf = MyDecisionTreeClassifier()
clf.fit(Train_X, Train_Y)
clf.print_tree()