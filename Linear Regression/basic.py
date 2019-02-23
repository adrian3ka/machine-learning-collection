
# coding: utf-8

# # Linear Regression

# In[1]:


import numpy as np
import pandas as pd
from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter


# In[2]:


get_ipython().magic(u'matplotlib inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')


# ### (Multiple) Linear Regression on the enrollment data

# In[3]:


file_path = './enrollment_forecast.csv'
enroll = pd.read_csv(file_path)

enroll.columns = ['year', 'roll', 'unem', 'hgrad', 'inc']
#roll -> enrollment
#unem -> unemployment
#hgrad -> high school graduation

enroll.head()


# In[4]:


sb.pairplot(enroll)
#make sure the variable have linear relationship


# In[5]:


print enroll.corr()


# In[6]:


enroll_data = enroll.ix[:,(2,3)].values
enroll_target = enroll.ix[:,1].values

enroll_data_names = ['unem', 'hgrad']
X, y = scale(enroll_data), enroll_target


# ### Checking for missing values

# In[9]:


missing_values = X == np.NAN
X[missing_values == True]


# In[12]:


LinReg = LinearRegression(normalize=True)

LinReg.fit(X,y)

print LinReg.score(X,y)

