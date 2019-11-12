#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from sklearn import datasets
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics


# In[4]:


dataset=pd.read_csv('Simple_Linear_Regression_Sales_data.csv')
df = pd.DataFrame(dataset)
df.head()


# In[5]:


dataset.keys()


# In[7]:


dataset.shape


# In[10]:


x, y = dataset.TV, dataset.Sales

print(x.head())
x = dataset.iloc[:,:-1].values
print(x[:5])


print(y.head())
y = dataset.iloc[:,1].values
print(y[:5])


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)


# In[12]:


x_test.shape


# In[13]:


regressor = LinearRegression()


# In[14]:


regressor.fit(x_train, y_train)


# In[15]:


y_predictions = regressor.predict(x_test)


# In[58]:


y_predictions[0]


# In[59]:


y_test[0]


# In[21]:


plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train))


# In[63]:


r2_score(y_test, y_predictions)


# In[ ]:




