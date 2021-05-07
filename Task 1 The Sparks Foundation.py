#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation GRIP Task-1, Prediction Using Supervised Machine Learning
# 
# # Author- Neha Panwar
# 
# AIM - Predict the percentage of an student based on the number of study hours. What will be the predicted score if a student studies for 9.25 hrs/day?

# # Importing Relevant Libraries

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# # Importing Dataset

# In[2]:


data= pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
print("Data has been successfully imported")
data


# In[3]:


data.describe()


# # Data Preprocessing

# # Distribution Plots

# In[4]:


sns.distplot(data["Hours"])


# In[5]:


sns.distplot(data["Scores"])


# Thus we can conclude that there are no major outliers in the dataset.

# In[6]:


# Checking for null values. True denotes the null values and false denotes the non-null values.
data.isnull()


# In[7]:


# Checking missing values 
data.isnull().sum()


# There are no missing values in our dataset.

# # Visualising the data

# In[8]:


plt.scatter(data["Hours"], data["Scores"])
plt.title("Hours Vs Percentage")
plt.xlabel("Hours of Study")
plt.ylabel("Percentage")
plt.show()


# The scatter plot shows that there exists a positive linear relationship between percentage and hours of study. Thus a linear function would be the correct functional form for the model.

# # Specifying the model

# In[9]:


# Specifying dependent and independent variable
y= data["Scores"]
x1= data["Hours"]


# In[10]:


# Splitting the data into training and testing data 
y_train, y_test, x_train, x_test = train_test_split(y,x1, test_size= 0.2, random_state=42)

# Fitting the linear regression line
x=sm.add_constant(x_train)
linear_regression_model = sm.OLS(y_train,x).fit()


# In[11]:


# Summary of the linear regression
linear_regression_model.summary()


# We can conclude from the given statistics that number of hours spent on study is highly significant at any chosen level of significance and explains approximately 95% of the variation in scores.

# # Plotting the regression line

# In[12]:


plt.scatter(x_train, y_train)
y_hat = x_train*9.6821 + 2.8269
plt.plot(x_train, y_hat, c="Orange")
plt.title("Hours Vs Percentage")
plt.xlabel("Hours of Study")
plt.ylabel("Scores")
plt.show()


# # Testing the model

# In[13]:


df = pd.DataFrame({"Constant":1,"Hours of Study": x_test})
prediction = linear_regression_model.predict(df)
Results = pd.DataFrame({"Actual Score":y_test, "Predicted Score": prediction})
print("Results")
Results = Results.reset_index(drop=True)
Results   


# In[14]:


plt.scatter(x_test, y_test)
y_hat = x_train*9.6821 + 2.8269
plt.plot(x_train, y_hat, c="Orange")
plt.title("Hours Vs Percentage")
plt.xlabel("Hours of Study")
plt.ylabel("Scores")
plt.show()


# # Predicted Score

# In[15]:


z= linear_regression_model.predict([1,9.25])
df= pd.DataFrame([9.25], columns=["Hours of Study/day"])
df["Predicted score"]=z
df


# # Conclusion
# 
# The predicted score if a student studies for 9.25 hrs/day is 92.386.
