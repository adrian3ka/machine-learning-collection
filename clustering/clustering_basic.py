
# coding: utf-8

# # K-means method

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = 7,4


# In[20]:


iris = datasets.load_iris()
print iris.data[0:10,]
X = scale(iris.data)
print X[0:10,]
y = pd.DataFrame(iris.target)
variable_names = iris.feature_names

X[0:10,]


# # Building and running your model

# In[90]:


clustering = KMeans(n_clusters = 3, random_state=5)
clustering.fit(X)


# In[111]:


print clustering.labels_

#Label ini tidak menunjukkan dia termasuk kategori mana, namun menunjukkan dia masuk ke cluster mana pada saat kalkulasi.
#Maka dari itu nanti perlu untuk di relabel


# # Plotting your model outputs

# In[112]:


iris_df = pd.DataFrame(iris.data)
print iris_df
iris_df.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

y.columns = ['Targets']


# In[113]:


color_theme = np.array(['darkgray','lightsalmon','powderblue'])


# In[118]:


plt.subplot(1,2,1)
plt.scatter(x= iris_df.Petal_Length.values,
            y=iris_df.Petal_Width.values,
            c = color_theme[iris.target],
            s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,
            y=iris_df.Petal_Width, 
            c = color_theme[clustering.labels_], 
            s=50)
plt.title('K-Means Classification')


# In[116]:


print clustering.labels_
print iris.target

#ingat, karena clustering.labels_ tidak menunjukkan kategori maka perlu untuk di relabel.


# In[119]:


relabel = np.choose(clustering.labels_, [2,0,1]).astype(np.int64)
print relabel
plt.subplot(1,2,1)
plt.scatter(x= iris_df.Petal_Length.values,
            y=iris_df.Petal_Width.values,
            c = color_theme[iris.target],
            s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,
            y=iris_df.Petal_Width, 
            c = color_theme[relabel], 
            s=50)
plt.title('K-Means Classification')


# # Evaluate your clustering results

# In[42]:


print (classification_report(y, relabel))


# In[101]:


target_index = 5

print X[target_index]
print clustering.labels_[target_index]
print iris.target[target_index]

data = np.array(X[target_index]) #ingat tadi 1 direlabel jadi 0
# 0 => 2
# 1 => 0
# 2 => 1
data.reshape(1, -1)

print clustering.predict(data.reshape(1,-1))


# In[120]:


clustering.cluster_centers_

