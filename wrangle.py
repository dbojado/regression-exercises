#!/usr/bin/env python
# coding: utf-8

# # Wrangle
# 
# We will start with mall_customers database.
# 
# - acquire verify our acquire module is working 
# - summarize our data
# - plot histograms and boxplots
# - na's
# - outliers
# - astype()
# - pd.cut()

# # Practice with Mall Data

# # Acquire

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import env 
import os

import acquire


# In[2]:


df = acquire.get_mall_data()


# # Summarize

# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.dtypes


# # Takeaways

# - encode gender
# - drop customer_id
# - define our target variable: spending_score

# In[6]:


df.gender.value_counts()


# In[7]:


df.info()


# In[8]:


df.describe()


# Takeaways
# - not sure how annual income is computed or what units it is in
# - don't know what spending score means, assuming that the higher the score, more likely to spend...better to market to?

# # Plot distributions

# Histograms

# In[9]:


for col in ['age', 'annual_income', 'spending_score']:  
    plt.hist(df[col])
    plt.title(col)
    plt.show()


# # Boxplots

# In[10]:


sns.boxplot(data=df[['age','annual_income','spending_score']])


# nulls, outliers, cut, dropma, fillna, replace, get_dummies
# 
# train, validate, test split
# update prepare.py file to make necessary changes

# In[11]:


df.isna()


# This gives a true or false for every item in the dataframe

# In[12]:


df.isna().sum()


# **Takeaway:** There are no nulls in our data set  

# **Cutting / Binning** lets us turn a continous variable into a categorical one.  
# 
# - pd.cut creates bins of equal width
# - pd.qcut creates bins with the same number of observatons in each   
# 
# This is can be useful for initial exploration, interpretation, and visualization.

# In[13]:


get_ipython().run_line_magic('pinfo', 'pd.cut')


# In[14]:


pd.cut(df.age, bins=4).value_counts()


# In[15]:


pd.qcut(df.age, 4).value_counts().sort_index()


# In[16]:


pd.cut(df.age, bins=[0,30,50,100]).value_counts().sort_index()


# In[17]:


df['is_female'] = (df.gender == 'Female').astype('int')
df.head()


# In[18]:


from sklearn.model_selection import train_test_split

train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
train, validate = train_test_split(train_and_validate, test_size=.15, random_state=123)

print('train', train.shape)
print('test', test.shape)
print('validate', validate.shape)


# # Workflow
# 
# - notebook -> python script workflow
#     - start out in a notebook
#     - experiment and rapidly iterate
#     - consolidate code and move into a .py script
#     - import the .py script into our notebook 
# - data flow: acquire - prepare - exploration
# - what are the benefits of a py script over a notebook?
#     - so we dont have to use jupyter notebook
#     - easier to transfer info
#     - keep notebook for insights 
#     - we can import functions from py scripts
#     - better project orgnanization 
# - What ae some downsides of py files? 
#     - harder to interpret comments
#     - no kernels, the whole script has to be run at once
# - markdown
#     - "y" turns it into a code cell  
#     - "m" turns it into a markdown cell

# Grab all the code that modifies dataframe

# In[19]:


# initial code
df = acquire.get_mall_data()

# modification to code
def prep_mall_data(df):
    '''
    Takes the acquired mall data, does data prep, and returns 
    train, test, and validate data splits.
    '''
    df['is_female'] = (df.gender == 'Female').astype('int')
    train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.15, random_state=123)
    return train, test, validate


# # Telco Churn Exercise Problem

# Throughout the exercises for Regression in Python lessons, you will use the following example scenario: As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.

# - The first step will be to acquire and prep the data. 

# In[20]:


df = acquire.get_telco_data()


# In[21]:


df.shape


# In[22]:


df.head() 


# In[23]:


df.dtypes


# In[24]:


df.isna().sum()


# ### 1. Acquire customer_id, monthly_charges, tenure, and total_charges from telco_churn database for all customers with a 2 year contract.

# In[25]:


def get_connection(db, username=env.username, host=env.host, password=env.password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

def new_telco_data():
    '''
    This function reads the telco churn data from the Codeup db into a df, 
    write it to a csv file, and returns the df.
    '''
    sql_query = 'SELECT customer_id, monthly_charges, tenure, total_charges FROM customers WHERE contract_type_id = 3';
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    df.to_csv('telco_churn_df.csv')
    return df

def get_telco_data(cached=False):
    '''
    This function reads in telco churn data from Codeup database if cached == False
    or if cached == True reads in telco churn df from a csv file, returns df
    '''
    if cached or os.path.isfile('telco_churn_df.csv') == False:
        df = new_telco_data()
    else:
        df = pd.read_csv('telco_churn_df.csv', index_col=0)
    return df


# In[26]:


df = new_telco_data()


# In[27]:


df.head()


# ### 2. Walk through the steps above using your new dataframe. You may handle the missing values however you feel is appropriate.

# In[28]:


df.shape


# In[29]:


df.dtypes


# In[30]:


df.describe()


# In[31]:


# remove white space
df['total_charges'] = df.total_charges.where((df.tenure != 0),0)

# convert data type to float
df['total_charges'] = df.total_charges.astype(float)


# In[32]:


df.dtypes


# In[33]:


df.isna().sum()


# In[34]:


# Histograms


# In[35]:


for col in ['monthly_charges', 'tenure', 'total_charges']:  
    plt.hist(df[col])
    plt.title(col)
    plt.show()


# In[36]:


# Boxplots


# In[37]:


sns.boxplot(data=df[['monthly_charges', 'tenure', 'total_charges']])


# <b> Takeaways: </b>
# - no missing values 
# - converted total_charges into a float value 

# In[38]:


# split the data into train, validate and test
train_validate, test = train_test_split(df, test_size=.2, random_state=123)
train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

train.shape, validate.shape, test.shape


# ### 3. End with a python file wrangle.py that contains the function, wrangle_telco(), that will acquire the data and return a dataframe cleaned with no missing values.

# In[39]:


def wrangle_telco():
    df = get_telco_data()
    df['total_charges'] = df.total_charges.where((df.tenure != 0),0)
    df['total_charges'] = df.total_charges.astype(float)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    return train, validate, test


# In[40]:


train, validate, test


# In[41]:


train.shape


# <b> Takeaways: </b>
# - filtered data 
