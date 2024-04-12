#!/usr/bin/env python
# coding: utf-8

# # Employee Attrition Analysis and Prediction

# # CONTENTS :
# 

# # EDA
# 1. Data Exploration.
# 2. Data Cleaning.
# 3. Data Encoding.
# 4. Data Labelling.
# 

# # Importing librarys

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


# # You've imported the necessary libraries for data manipulation (pandas), numerical operations (numpy), and data visualization (matplotlib.pyplot and seaborn). These libraries provide various functions and tools to work with data efficiently and visualize it effectively.

# In[2]:


# Read the dataset
df = pd.read_csv(r'F:\Technocolabs\WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[3]:


df


# # EDA

# # Data Cleaning.

# In[4]:


df.head()


# In[27]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[10]:


df.describe()


# In[28]:


df.dropna(inplace=True)
print(df)


# In[29]:


# Check the data types after conversion
print(df.dtypes)


# # Ensuring the dataset is clean and ready for analysis is crucial. You've checked for missing values using df.isnull().sum() and found that there are no missing values in the dataset. This suggests that there is no need to handle missing data.

# # Data Exploration:

# In[11]:


# Value counts for categorical variables
print(df['MonthlyRate'].value_counts())


# In[12]:


print(df['DailyRate'].value_counts())


# In[13]:


print(df['BusinessTravel'].value_counts())


# In[14]:


print(df['Department'].value_counts())


# In[15]:


print(df['EducationField'].value_counts())


# In[16]:


print(df['Gender'].value_counts())


# In[17]:


print(df['JobRole'].value_counts())


# In[18]:


print(df['MaritalStatus'].value_counts())


# In[19]:


print(df['Over18'].value_counts())


# In[20]:


print(df['OverTime'].value_counts())


# # This involves getting a general understanding of the dataset. You've used df.shape to check the dimensions (number of rows and columns) of the dataset, df.columns to see the column names, and df.head() to display the first few rows of the dataset. These steps help you understand the structure and contents of the data.

# # Data Encoding

# In[31]:


if 'categorical_column' in df.columns:
    # One-hot encoding
    encoded_df = pd.get_dummies(df, columns=['categorical_column'])
    print(encoded_df.head())
else:
    print("The column 'categorical_column' does not exist in the DataFrame. Please provide the correct column name.")


# In[32]:


# One-hot encoding
encoded_df = pd.get_dummies(df, columns=['Department'])


# In[33]:


# Label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['encoded_column'] = label_encoder.fit_transform(df['Department'])


# In[34]:


# Specify the list of categorical columns you want to one-hot encode
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

# One-hot encoding
encoded_df = pd.get_dummies(df, columns=categorical_columns)
print(encoded_df.head())


# This encoding technique transforms categorical variables into a format suitable for machine learning algorithms, facilitating the analysis and modeling of categorical data.

# In[35]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode a specific column
df['Attrition_encoded'] = label_encoder.fit_transform(df['Attrition'])
print(df[['Attrition', 'Attrition_encoded']].head())


# The provided code utilizes the LabelEncoder from scikit-learn to encode the 'Attrition' column in the DataFrame:
# 
# Label Encoding: The LabelEncoder is initialized to transform categorical labels into numerical values.
# 
# Encoding Process: The 'Attrition' column is encoded using the fit_transform method of the LabelEncoder, which assigns numerical labels to the categories.
# 
# Output: The code prints the first few rows of the DataFrame with both the original 'Attrition' column and the newly encoded 'Attrition_encoded' column.
# 
# This encoding process converts categorical data into a format suitable for machine learning algorithms that require numerical input, enabling further analysis and modeling.
# 
# 
# 
# 
# 

# In[36]:


from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Scale numerical features
numerical_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[37]:


from sklearn.model_selection import train_test_split

# Split data into features (X) and target variable (y)
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Encoding categorical variables into numerical format is necessary for many machine learning algorithms. However, in the code you provided, it seems like the dataset doesn't contain categorical variables that need encoding. If there were categorical variables, you might use techniques like one-hot encoding or label encoding to convert them into numerical format.

# # Data Labelling

# In[53]:


def transform(feature):
    le=LabelEncoder()
    df[feature]=le.fit_transform(df[feature])
    print(le.classes_)



# In[54]:


cat_df=df.select_dtypes(include='object')
cat_df.columns


# In[55]:


for col in cat_df.columns:
    transform(col)


# In[38]:


# Replace 'your_dataset.csv' with the actual path to your dataset file
df = pd.read_csv(r'F:\Technocolabs\WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Define your condition and label accordingly
def label_function(row):
    if row['Age'] > 30:
        return 'label_A'
    else:
        return 'label_B'

# Apply the label function to each row
df['label'] = df.apply(label_function, axis=1)

# Display the first few rows to verify the labeling
print(df.head())


# This labeling process enables segmentation and analysis of the dataset based on age categories, providing insights into workforce demographics and potential age-related patterns or trends.

# In[39]:


# Define your conditions and labels accordingly
def label_function(row):
    if row['Age'] > 30 and row['Department'] == 'Sales':
        return 'label_A'
    # Example: If DailyRate is less than 500 and Education is greater than 3, label as 'label_B'
    elif row['DailyRate'] < 500 and row['Education'] > 3:
        return 'label_B'
    # Add more conditions and labels as needed
    else:
        return 'label_C'  # Default label if none of the conditions are met

# Apply the label function to each row
df['label'] = df.apply(label_function, axis=1)

# Display the first few rows to verify the labeling
print(df.head())


# The labeling function categorizes employees into different groups based on age, department, daily rate, and education level, allowing for insights into specific employee demographics and characteristics.
# This segmentation can aid in identifying patterns or trends within the workforce, such as the distribution of older employees in the Sales department ('label_A'), or the prevalence of employees with lower daily rates and higher education levels ('label_B'), providing valuable insights for targeted HR strategies or organizational decision-making.
# 
# 
# 
# 
# 
# 

# In[40]:


# Define your conditions and labels accordingly
def label_function(row):
    if row['Age'] > 30 and row['Department'] == 'Sales':
        return 'label_A'
    # Example: If DailyRate is less than 500 and Education is greater than 3, label as 'label_B'
    elif row['DailyRate'] < 500 and row['Education'] > 3:
        return 'label_B'
    # Add more conditions and labels as needed
    else:
        return 'label_C'  # Default label if none of the conditions are met

# Apply the label function to each row
df['label'] = df.apply(label_function, axis=1)

# Display the first few rows to verify the labeling
print(df.head())


# The provided code applies a labeling function to categorize employees in the dataset based on specific conditions:
# 
# Labeling Conditions: Employees are labeled as 'label_A' if they are over 30 years old and belong to the Sales department, 'label_B' if their daily rate is below 500 and education level is greater than 3, and 'label_C' otherwise.
# 
# Applying the Labeling Function: The function is applied to each row of the DataFrame, resulting in a new column named 'label' containing the assigned labels.
# 
# These labeled categories enable further analysis and segmentation of the dataset, providing insights into employee demographics and characteristics based on predefined criteria.
# 
# 
# 
# 
# 
# 

# In[41]:


# Define your conditions and labels accordingly
def label_function(row):
    if row['Age'] > 30 and row['Department'] == 'Sales':
        return 'label_A'
    # Example: If DailyRate is less than 500 and Education is greater than 3, label as 'label_B'
    elif row['DailyRate'] < 500 and row['Education'] > 3:
        return 'label_B'
    # Add more conditions and labels as needed
    else:
        return 'label_C'  # Default label if none of the conditions are met

# Apply the label function to each row
df['label'] = df.apply(label_function, axis=1)

# Display the first few rows to verify the labeling
print(df.head())


# 
# The labeling function categorizes employees in the dataset based on conditions related to age and department or daily rate and education level, assigning them labels 'label_A', 'label_B', or 'label_C' for further analysis and segmentation.
# 
# 
# 
# 
# 

# In[42]:


# Define your conditions and labels accordingly
def label_attrition(row):
    if row['Attrition'] == 'Yes':
        return 'High Attrition'
    else:
        return 'Low Attrition'

def label_income(row):
    if row['MonthlyIncome'] > 5000:
        return 'High Income'
    else:
        return 'Low Income'

# Apply the label functions to each row
df['Attrition_Label'] = df.apply(label_attrition, axis=1)
df['Income_Label'] = df.apply(label_income, axis=1)

# Display the first few rows to verify the labeling
print(df.head())


# The labeled data facilitates insights into workforce dynamics by categorizing employees based on attrition and income levels, enabling targeted analysis for understanding retention challenges and income distribution patterns.

# # Labeling the data based on certain conditions or criteria can be useful for various analyses. In this case, it seems like the dataset doesn't require explicit labeling based on the provided code snippet.

# # Visualization Let us first analyze the various numeric features.

# In[21]:


# Visualization
sns.histplot(df['Age'])
plt.show()


# Visualizing the age distribution through a histogram provides insights into the workforce's central tendency, spread, skewness, outliers, and age composition, aiding in demographic understanding and HR decision-making.
# 
# 
# 
# 
# 
# 

# In[7]:


# Plotting a box plot for MonthlyIncome
plt.figure(figsize=(10, 6))
plt.boxplot(data['MonthlyIncome'])
plt.title('Box plot of Monthly Income')
plt.ylabel('Monthly Income')
plt.show()


# Note that all the features have pretty different scales and so plotting a boxplot is not a good idea. Instead what we can do is plot histograms of various continuously distributed features.

# In[22]:


# Visualization
plt.figure(figsize=(10, 5))

# Histogram for Monthly Income
plt.subplot(1, 2, 1)
sns.histplot(df['MonthlyIncome'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Monthly Income')

# Histogram for Total Working Years
plt.subplot(1, 2, 2)
sns.histplot(df['TotalWorkingYears'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Total Working Years')

plt.tight_layout()
plt.show()


# This code creates histograms to visually compare the distributions of monthly income and total working years, offering insights into the income and tenure composition of the workforce.

# In[23]:


# Visualization
plt.figure(figsize=(10, 6))

# Line plot for change in Monthly Income across Age
sns.lineplot(x='Age', y='MonthlyIncome', data=df, ci=None)
plt.title('Change in Monthly Income Across Age')
plt.xlabel('Age')
plt.ylabel('Monthly Income')

plt.tight_layout()
plt.show()


# This code generates a line plot illustrating the change in monthly income across different ages.
# 
# Insight from this visualization:
# The line plot reveals any trends or patterns in how monthly income varies with age, providing insights into potential age-related income progression or stagnation within the workforce.

# In[24]:


# Visualization
plt.figure(figsize=(10, 6))

# Histogram for Total Working Years
sns.histplot(df['TotalWorkingYears'], bins=20, kde=True)
plt.title('Distribution of Total Working Years')
plt.xlabel('Total Working Years')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# This code produces a histogram to visualize the distribution of total working years among employees.
# 
# Insight from this visualization:
# The histogram provides an overview of the frequency of different total working year intervals within the workforce, offering insights into the distribution of employee tenure and potential patterns in work experience accumulation.

# In[25]:


# Visualization
plt.figure(figsize=(10, 6))

# Histogram for Total Working Years
sns.histplot(df['TotalWorkingYears'], bins=20, kde=True, color='skyblue', edgecolor='black', linewidth=1.2)
plt.title('Distribution of Total Working Years')
plt.xlabel('Total Working Years')
plt.ylabel('Frequency')

# Adding grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adding mean and median lines
mean_total_working_years = df['TotalWorkingYears'].mean()
median_total_working_years = df['TotalWorkingYears'].median()
plt.axvline(mean_total_working_years, color='red', linestyle='--', label=f'Mean: {mean_total_working_years:.2f}')
plt.axvline(median_total_working_years, color='green', linestyle='--', label=f'Median: {median_total_working_years}')

# Adding legend
plt.legend()

plt.tight_layout()
plt.show()


# This code enhances the visualization of the distribution of total working years by adding features such as color, gridlines, and lines indicating the mean and median values.
# 
# Insights from this visualization:
# 
# The histogram displays the frequency of different total working year intervals, with the KDE (Kernel Density Estimation) curve providing a smoothed estimate of the distribution.
# The gridlines improve readability, making it easier to interpret the distribution.
# The red dashed line represents the mean total working years, while the green dashed line represents the median total working years, providing key summary statistics for the distribution.
# The legend helps in identifying the meaning of the dashed lines.
# Overall, this visualization offers a comprehensive view of the distribution of total working years within the workforce, highlighting central tendency measures and distribution characteristics.
# 
# 
# 
# 
# 
# 

# In[26]:


# Additional Visualization
plt.figure(figsize=(10, 6))

# Scatter plot: Age vs. Monthly Income
sns.scatterplot(data=df, x='Age', y='MonthlyIncome', hue='Attrition', palette='Set1', alpha=0.8)
plt.title('Age vs. Monthly Income')
plt.xlabel('Age')
plt.ylabel('Monthly Income')

# Adding grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adding legend
plt.legend(title='Attrition')

plt.tight_layout()
plt.show()


# This additional visualization is a scatter plot illustrating the relationship between age and monthly income, with points differentiated by attrition status.
# 
# Insights from this visualization:
# 
# The scatter plot helps identify any patterns or trends in how monthly income varies with age.
# Points are color-coded based on attrition status, allowing for the comparison of income-age dynamics between employees who have churned (attrition = Yes) and those who haven't (attrition = No).
# The gridlines enhance readability, aiding in the interpretation of data points.
# The legend clarifies the meaning of different colors in the plot, distinguishing between employees who have left the company and those who haven't.
# Overall, this visualization offers insights into the relationship between age, monthly income, and attrition, potentially highlighting age-related attrition patterns or income disparities within the workforce.
# 
# 
# 
# 
# 
# 

# We can also plot a kdeplot showing the distribution of the feature. Below I have plotted a kdeplot for the 'Age' feature. Similarly we plot for other numeric features also. We can also use a distplot from seaborn library.

# In[48]:


sns.kdeplot(df['Age'],shade=True,color='#ff4125')


# In[49]:


sns.distplot(df['Age'])


# I have made a function that accepts the name of a string. In our case this string will be the name of the column or attribute which we want to analyze. The function then plots the countplot for that feature which makes it easier to visualize.

# # Let us now similalry analyze other categorical features.

# In[59]:


import seaborn as sns

def plot_cat(column_name):
    sns.catplot(x=column_name, kind='count', data=df)

# Now you can call the function with the column name as an argument
plot_cat('Attrition')


# In[60]:


plot_cat('BusinessTravel')


# In[61]:


plot_cat('OverTime')


# In[62]:


plot_cat('Department')


# In[63]:


plot_cat('EducationField')


# Note that the same function can also be used to better analyze the numeric discrete features like 'Education' ,'JobSatisfaction' etc...

# In[64]:


plot_cat('JobRole')


# Note that the number of observations belonging to the 'No' category is way greater than that belonging to 'Yes' category. Hence we have skewed classes and this is a typical example of the 'Imbalanced Classification Problem'. To handle such types of problems we need to use the over-sampling or under-sampling techniques. I shall come back to this point later.
# 

# # Visualizing the data helps in gaining insights and understanding the relationships between different variables. In your code, you've created several visualizations:
# Histograms (df.hist()) to visualize the distribution of numerical variables. Histograms provide a graphical representation of the frequency distribution of data.
# Box plots (sns.boxplot()) to identify outliers and understand the distribution of numerical variables. Box plots display the distribution of data based on quartiles and help in detecting potential anomalies.
# Pair plots (sns.pairplot()) to visualize pairwise relationships between different variables in the dataset. Pair plots are useful for identifying patterns and correlations between variables.
# 

# # Crosstabulation of Attrition

# In[67]:


pd.crosstab(columns=[df.Attrition],index=[df.JobLevel],margins=True,normalize='index') # set normalize=index to view rowwise %.


# In[68]:


pd.crosstab(columns=[df.Attrition],index=[df.JobSatisfaction],margins=True,normalize='index') # set normalize=index to view rowwise %.


# In[69]:


pd.crosstab(columns=[df.Attrition],index=[df.EnvironmentSatisfaction],margins=True,normalize='index') # set normalize=index to view rowwise %.


# In[70]:


pd.crosstab(columns=[df.Attrition],index=[df.JobInvolvement],margins=True,normalize='index') # set normalize=index to view rowwise %.


# Note this shows an interesting trend. Note that for higher values of job satisfaction( ie more a person is satisfied with his job) lesser percent of them say a 'Yes' which is quite obvious as highly contented workers will obvioulsy not like to leave the organisation.

# # conclusion

# # Overall, the code snippet provided shows the initial steps of data exploration and visualization, including checking data integrity, understanding variable distributions, and visualizing relationships between variables. These steps are essential for gaining insights into the data and informing subsequent analysis and modeling tasks

# In[ ]:




