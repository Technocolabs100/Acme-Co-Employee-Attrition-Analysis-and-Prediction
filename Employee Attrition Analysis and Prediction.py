#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


data = pd.read_csv('Dataset - HR-Employee-Attrition.csv')


# In[9]:


data.head()


# In[10]:


pd.set_option('display.max_columns',None)


# In[11]:


data.head()


# In[12]:


data.info()


# In[13]:


data.describe()


# In[14]:


print(data.duplicated().value_counts())
data.drop_duplicates(inplace=True)
print(len(data))


# In[15]:


data.isnull().sum()


# In[16]:


plt.figure(figsize=(12,5))
sns.countplot(x='Department',hue='Attrition',data=data,palette='plasma')
plt.title("Attrition per Department")
plt.show()


# In[17]:


plt.figure(figsize=(12,5))
sns.countplot(x='EducationField',hue='Attrition',data=data,palette='plasma')
plt.title("Attrition per Education Field")
plt.show()


# In[18]:


plt.figure(figsize=(12,5))
sns.countplot(x='Gender',hue='Attrition',data=data,palette='hot')
plt.title("Attrition per Gender")
plt.show()


# In[19]:


data['count'] = 1


# In[20]:


data.groupby(["Gender", "Attrition"]).agg({"count":"sum"})


# Observations : - Employees from Sales departement are more likely to leave the job early.
# Males Attrition is higher than Females's.

# In[21]:


numerical_features = ['Education','DistanceFromHome','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance','YearsSinceLastPromotion','YearsWithCurrManager']
data[numerical_features].head()


# Label Encoding

# In[22]:


data['Attrition'] = data['Attrition'].replace({'No':0,'Yes':1})


# In[23]:


data['OverTime'] = data['OverTime'].map({'No':0,'Yes':1})
data['Gender'] = data['Gender'].map({'Male':0,'Female':1})
data['Over18'] = data['Over18'].map({'Y':0,'No':1})


# In[24]:


from sklearn.preprocessing import LabelEncoder
encod = ['BusinessTravel','Department','EducationField','JobRole','MaritalStatus']
label_encoders = {}
for column in encod : 
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])


# In[25]:


data.head()


# In[26]:


data.info()


# In[33]:


financial_features = ['OverTime','EnvironmentSatisfaction','Gender','MaritalStatus']


# In[34]:


fig=plt.subplots(figsize=(10,15))

for p,q in enumerate(financial_features):
    plt.subplot(4,2,p+1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=q,data=data)


# In[40]:


plt.figure(figsize=(12,5))
sns.kdeplot(data['Age'],shade=True)
plt.title('Employees by Age')
plt.xlabel('Age')
plt.ylabel('count')
plt.show()


# Most employees are aged between 25 and 45

# In[49]:


plt.figure(figsize=(12, 5))
sns.countplot(x='Education', hue='Attrition', data=data, palette='coolwarm') 
plt.title("Education vs Attrition")
plt.show()


# Bivariate Analysis

# In[52]:


plt.figure(figsize=(8, 6))
plt.scatter(data['Age'], data['MonthlyIncome'])
plt.xlabel('Age')
plt.ylabel('MonthlyIncome')
plt.title('Bivariate Analysis of Age vs Salary')
plt.show()


# In[54]:


#Heatmap using Seaborn
correlation_matrix = data[['Age', 'MonthlyIncome', 'Education', 'Attrition']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Multivariate Analysis

# In[57]:


plt.figure(figsize=(10, 8))
sns.barplot(x='Department', y='Attrition', hue='EducationField', data=data)
plt.title('Grouped Bar Plot of Salary by Department and Education Field')
plt.show()


# In[ ]:




