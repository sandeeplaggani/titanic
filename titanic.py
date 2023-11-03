#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns


# In[24]:


# Load the Titanic dataset
df = pd.read_csv("train titanic.csv")


# In[25]:


df.info()


# In[26]:


df.isnull()


# In[27]:


df.dropna()


# In[28]:


df.describe()


# In[29]:


df.fillna(value="nan")


# In[30]:


df.head()


# In[31]:


sns.pairplot(df,hue="Sex")


# # Data Preprocessing

# In[32]:


df= df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df = df.dropna()


# In[33]:


from sklearn.model_selection import train_test_split


# In[36]:


X =df.drop('Survived', axis=1)
y = df['Survived']


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[38]:


from sklearn.ensemble import RandomForestClassifier


# In[39]:


# Train a Random Forest Classifier
rc = RandomForestClassifier(n_estimators=100, random_state=42)
rc.fit(X_train, y_train)


# In[41]:


y_pred = rc.predict(X_test)


# In[42]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[43]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[44]:


print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')


# In[ ]:




