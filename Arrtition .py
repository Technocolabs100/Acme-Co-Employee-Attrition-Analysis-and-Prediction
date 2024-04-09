#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv(r'C:\Users\Lenovo\Downloads\WA_Fn-UseC_-HR-Employee-Attrition\WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[3]:


data.head()


# In[4]:


pd.set_option('display.max_columns',None)


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


# Observations 
# we only have int and string data types features. there is no feature with float. 26 features are numerical and 9 features are caytegorical
# Attrition is our target value which has no missing values. But the quantity of data of emp having Attrition is less compared to employees which do not habe Attrition
# There is no missing value in dataset


# In[9]:


print(data.duplicated().value_counts())
data.drop_duplicates(inplace = True)
print (len(data))


# In[10]:


data.isnull().sum()


# In[11]:


plt.figure(figsize=(15,5))
plt.rc("font", size=14)
sns.countplot(y = 'Attrition',data=data)
plt.show()


# In[12]:


# Over here we notice that the Target column is Highly imbalanced, we need to balance the data by using some Statistical Methods.


# In[13]:


# Department wrt Attrition
plt.figure(figsize=(12,5))
sns.countplot(x='Department',hue='Attrition',data=data, palette='hot')
plt.title("Attrition w.r.t Department")
plt.show()


# In[14]:


# Education wrt Attrition
plt.figure(figsize=(12,5))
sns.countplot(x='EducationField', hue='Attrition', data=data, palette='hot')
plt.title("Attrition w.r.t EducationField")  # Corrected function name
plt.xticks(rotation=45)
plt.show()


# In[15]:


# Jobrole wrt Attrition
plt.figure(figsize=(12, 5))
sns.countplot(x='JobRole', hue='Attrition', data=data, palette='hot') 
plt.title("JobRole w.r.t Attrition")
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.show()


# In[16]:


#Gender wrt Attrition
plt.figure(figsize=(12, 5))
sns.countplot(x='Gender', hue='Attrition', data=data, palette='hot')  
plt.title("Gender w.r.t Attrition")
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.show()


# In[17]:


# Observations
# Employees working in R&D department are more, but employees from sales department leaves the job early.
# Males are more under Attrition than Females.


# In[18]:


# Age distribution 
plt.figure(figsize=(12,5))
sns.distplot(data['Age'], kde=False)
plt.show()


# In[19]:


# Age column is very well normalized, most of employees are of age between 25 to 40.
# Dataset is having some of the numerical columns which are lebel encoded, they are ordinal labels.


# In[20]:


ordinal_features = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance']
data[ordinal_features].head()


# In[21]:


# Observation 
# Employees from Bachelor are more, then from Massters background. Attrition wrt to bachelor can be seen more.


# In[22]:


#Target Variale (Attrition)
data['Attrition'] = data['Attrition'].replace({'No':0,'Yes':1})


# In[23]:


#encode binary variables
data['OverTime'] = data['OverTime'].map({'No': 0, 'Yes': 1})
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})


# In[24]:


data['Over18'] = data['Over18'].map({'Y': 1, 'N': 0})


# In[25]:


#label encoder to df_categorical
from sklearn.preprocessing import LabelEncoder

encoding_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
label_encoders = {}

for column in encoding_cols:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])


# In[26]:


data.head()


# In[ ]:


# Resampling
# OverSampling


# In[28]:


from collections import Counter
from imblearn.over_sampling import RandomOverSampler

X = data.drop(columns=['Attrition'])  
y = data['Attrition']  

print("Class distribution before oversampling:", Counter(y))

rus = RandomOverSampler(random_state=42)
X_over, y_over = rus.fit_resample(X, y)

print("Class distribution after oversampling:", Counter(y_over))


# In[29]:


# Training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size = 0.2, random_state = 42)


# In[30]:


#Checking sample data 
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


#Logistic Regression


# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score


# In[34]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[35]:


prediction = logreg.predict(X_test)
cnf_matrix = confusion_matrix(y_test,prediction)
print("Accuracy Score-", accuracy_score(y_test , prediction))


# In[39]:


fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax1 = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues', fmt='d')
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)
plt.xlabel('Predicted')
plt.ylabel('Expected')

ax2 = fig.add_subplot(1, 2, 2)
y_pred_proba = logreg.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, prediction)
auc = roc_auc_score(y_test, prediction)
ax2 = plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:




