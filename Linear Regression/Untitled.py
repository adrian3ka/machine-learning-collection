
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20,10)


# In[2]:


#Reading pada
data = pd.read_csv('headbrain.csv')
print data.shape
data.head()


# In[3]:


#Collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values


# In[5]:


# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

#total number of values
m = len(X)

#using formula to calculate b1 and b0
numer = 0 
denom = 0 
for i in range (m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
    
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

#print coefficients
print b1, b0


# In[12]:


# Plotting Values and Regression Line

max_x = np.max(X) + 100
min_x = np.min(X) - 100

#Calculating line value X and Y
x = np.linspace(min_x, max_x, 1000)
y = b0 +  b1 * x

# Ploting line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X,Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Head Size in Cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()


# In[13]:


ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print r2


# # Using Sci-kit Learning

# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cannot use Rank 1 Matrix in scikit learn
X = X.reshape((m,1))

#Creating Model
reg = LinearRegression()

reg = reg.fit(X,Y)

Y_pred = reg.predict(X)

# Calculatring r2_score

r2_score = reg.score(X, Y)

print r2_score

