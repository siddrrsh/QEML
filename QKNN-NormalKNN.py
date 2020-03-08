#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# In[55]:


df = pd.read_csv("data 2.csv") 
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
    
print (df.head) 

df.info()

X=np.array(df.drop(['diagnosis'],1))
y=np.array(df['diagnosis'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)


# In[ ]:





# In[45]:


# define Malignant and Benign as 1 and 0 respectively
def diagnosis_value(diagnosis): 
    if diagnosis == 'M': 
        return 1
    else: 
        return 0
  
df['diagnosis'] = df['diagnosis'].apply(diagnosis_value) 


# In[46]:


# visualization
sns.lmplot(x = 'radius_mean', y = 'texture_mean', hue = 'diagnosis', data = df) 


# In[48]:


# do train-test split (2/3 to 1/3)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size = 0.33, random_state = 42) 


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13) 
knn.fit(X_train, y_train) 


# In[ ]:




