
# coding: utf-8

# In[2]:


from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_


# In[3]:


kmeans.predict([[0, 0], [4, 4]])


# In[4]:


kmeans.cluster_centers_

