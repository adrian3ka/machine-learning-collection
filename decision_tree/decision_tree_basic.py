
# coding: utf-8

# In[10]:


import pydotplus
import numpy as np


# In[11]:


from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image, display


# In[13]:


def load_data_set():
    """
    Loads the iris data set
 
    :return:        data set instance
    """
    iris = load_iris()
    print iris
    
    return iris


# In[14]:


def train_model(iris):
   """
   Train decision tree classifier

   :param iris:    iris data set instance
   :return:        classifier instance
   """
   print iris
   
   clf = tree.DecisionTreeClassifier()
   clf = clf.fit(iris.data, iris.target)
   return clf


# In[15]:


def display_image(clf, iris):
    """
    Displays the decision tree image
 
    :param clf:     classifier instance
    :param iris:    iris data set instance
    """
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True)
 
    graph = pydotplus.graph_from_dot_data(dot_data)
    display(Image(data=graph.create_png()))
 


# In[16]:


if __name__ == '__main__':
    iris_data = load_iris()
    decision_tree_classifier = train_model(iris_data)


# In[17]:


display_image(clf=decision_tree_classifier, iris=iris_data)


# In[30]:


data = np.array(iris_data.data[132])
print iris_data.data[132]
print data
print data.reshape(1,-1)
print iris_data.target[132]
data.reshape(1, -1)


# In[31]:


print decision_tree_classifier.predict(data.reshape(1,-1))

