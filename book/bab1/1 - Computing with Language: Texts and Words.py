
# coding: utf-8

# In[1]:


from nltk.book import *
from __future__ import division
import nltk
import numpy
import matplotlib

print('The nltk version is {}.'.format(nltk.__version__))


# In[2]:


text1.concordance("a")


# In[3]:


text2


# In[4]:


text1.concordance("monstrous")
text2.concordance("monstrous")


# In[5]:


text1.similar("monstrous")


# In[6]:


text2.similar("monstrous")


# In[7]:


text2.common_contexts(["monstrous", "very"])


# In[8]:


text1.concordance("affection")
text2.concordance("affection")


# In[9]:


text1.similar("affection")


# In[10]:


text2.similar("affection")


# In[11]:


text1.common_contexts(["affection","nearest"])


# In[19]:


text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])


# In[13]:


len(text3)


# In[20]:


sorted(set(text3))


# In[21]:


len(set(text3))


# In[22]:


len(text3) / len(set(text3))


# In[23]:


text3.count("smote")


# In[25]:


def lexical_diversity(text):
    return len(text)/len(set(text))
def percentage(count,total):
    return 100 * count / total


# In[26]:


lexical_diversity(text3)


# In[27]:


lexical_diversity(text5)


# In[29]:


percentage(text4.count('man'), len(text4))

