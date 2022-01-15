#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dt = pd.read_csv('heartdisease.csv')


# In[3]:


dt


# In[4]:


dt.info


# In[5]:


dt.isnull().sum()


# In[6]:


plt.scatter


# In[7]:


X = dt.drop(columns = 'target' , axis =1)
Y = dt['target']


# In[8]:


X


# In[9]:


Y


# In[10]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , Y, test_size = 0.2 , stratify = Y , random_state = 0)


# In[11]:


print(X_train.shape , X_test.shape)


# In[12]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[13]:


#Logistic
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train , y_train)


# In[14]:


y_pred = classifier.predict(X_test)


# In[15]:


y_pred


# In[16]:


from sklearn import metrics
print('accuracy' , metrics.accuracy_score(y_pred,y_test))


# In[18]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# 

# In[19]:


cm


# In[23]:


from matplotlib.colors import ListedColormap
X_set , y_set = X_train , y_train 
X1 , X2 = np.meshgrid(np.arrange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1 , step = 0.01 ,
                      np.arrange(start = X_set[: , 1].min() - 1, stop = X_set[:,1].max()+ 1 , step = 0.01))
plt.contourf(X1 , X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75 , cmap = ListedColormap(('red' ,'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(x2.min(), X2.max())
for i, j in enumerate(np.unique(y_set))
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,1],
                c = ListedColormap(('red' ,'green'))(i),label = j)
plt.title('Logistic Regression(Training set)')
plt.xlabel('Age')
plt.ylabel('Target')
plt.legend()
plt.show()
                      


# In[28]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski' , p=2)
classifier.fit(X_train, y_train)


# In[29]:


y_pred = classifier.predict(X_test)


# In[30]:


y_pred


# In[31]:


from sklearn import metrics
print('accuracy' , metrics.accuracy_score(y_pred,y_test))


# In[32]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# In[33]:


cm


# In[ ]:


#

