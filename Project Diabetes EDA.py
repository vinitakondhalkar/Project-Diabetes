#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize']=(12,6)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install missingno')
import missingno as msno
from sklearn.preprocessing import LabelEncoder


# In[2]:


df=pd.read_excel("Diabetes-Modified .xlsx")
dfDemographics=pd.read_excel("Diabetes-Modified .xlsx","Demographics in V2 and V8")
dfMedicationHistory=pd.read_excel("Diabetes-Modified .xlsx","Medical and Medication History ")
dfLabTests=pd.read_excel("Diabetes-Modified .xlsx","Lab Tests in V2 and V8")
dfMRI=pd.read_excel("Diabetes-Modified .xlsx","MRI Tests in V2 and V8")
dfCognitive=pd.read_excel("Diabetes-Modified .xlsx","Cognitive Tests")


# In[3]:


df.head(10)
df.shape


# In[4]:


dfDemographics.head(10)
dfDemographics.shape


# In[5]:


dfMedicationHistory.head(10)
dfMedicationHistory.shape


# In[6]:


dfLabTests.head(10)
dfLabTests.shape


# In[7]:


dfMRI.head(10)
dfMRI.shape


# In[8]:


dfCognitive.head(10)
dfCognitive.shape


# In[9]:


df.tail(5)


# In[10]:


(row,column) = df.shape


# In[11]:


row


# In[12]:


column


# In[13]:


#creates a tuple
df.shape


# In[14]:


df.info()
df.info(verbose=True,show_counts=True)


# In[15]:


df.describe(include='all')


# In[16]:


df.dtypes


# In[17]:


df.nunique()


# In[18]:


pd.set_option('max_rows', 99999)
pd.set_option('max_colwidth', 400)


# In[19]:


df.count()


# ************************* HANDLING MISSING VALUE - DEMOGRAPHICS ***********************

# In[20]:


#MISSING VALUE DEMOGRAPHICS change HTN or not column to mode value 
#modeHTN=df["HTN or not"].mode()
#modeHTN
dfDemographics['HTN or not'].fillna(dfDemographics['HTN or not'].mode()[0],inplace=True)


# In[21]:


dfDemographics['HTN or not'].value_counts()


# In[22]:


#df.iloc[:,0:4]
#df[df["Group"]=='CONTROL']
#MISSING VALUE DEMOGRAPHICS Diabetes Duration where group == control to 0 value 
dfDemographics.loc[dfDemographics['Group']=='CONTROL','Diabetes Duration']= 0
dfDemographics[{'Diabetes Duration','Group'}][dfDemographics["Group"]=='CONTROL']


# In[23]:


#MISSING VALUE DEMOGRAPHICS Diabetes Duration mean value 
meanDiabetesDuration = dfDemographics['Diabetes Duration'].mean()
meanDiabetesDuration
dfDemographics['Diabetes Duration'].replace(np.NaN,meanDiabetesDuration,inplace=True)


# In[24]:


#drop empty columns with all null values 
dfDemographics1=dfDemographics.dropna(how='all',axis=1)
dfDemographics1.isnull().sum()


# ************************* HANDLING MISSING VALUE - MEDICATION HISTORY ***********************

# In[25]:


dfMedicationHistory.shape #65 empty columns 

dfMedicationHistory1=dfMedicationHistory.dropna(how='all',axis=1)
dfMedicationHistory1.isnull().sum()

dfMedicationHistory1.shape #0 empty columns


# In[26]:


dfMedicationHistory1.sort_values(by='PatientID_x')
dfMedicationHistory1.value_counts(subset='PREVIOUS TOBACCO USE_x')


# In[27]:


le = LabelEncoder()
dfMedicationHistory1['PREVIOUS TOBACCO USE_x'] = le.fit_transform(dfMedicationHistory1['PREVIOUS TOBACCO USE_x'])
newdf=dfMedicationHistory1
newdf


# In[ ]:





# In[ ]:





# ************************* HANDLING MISSING VALUE - LAB TESTS ***********************

# In[28]:


dfLabTests1=dfLabTests.dropna(how='all',axis=1)
dfLabTests1.isnull().sum()


# ************************* HANDLING MISSING VALUE - MRI  ***********************

# In[29]:


dfMRI1=dfMRI.dropna(how='all',axis=1)
dfMRI1.isnull().sum()


# ************************* HANDLING MISSING VALUE - COGNITIVE ***********************

# In[30]:


dfCognitive1=dfCognitive.dropna(how='all',axis=1)
dfCognitive1.isnull().sum()


# MISSING NUMBERS LIBRARY 
# Python has a library named missingno which provides a few graphs that let us visualize missing data from a 
# different perspective. This can help us a lot in the handling of missing data.
# The missingno library is based on matplotlib.It has 4 plot as of now for the understanding distribution of missing data in our dataset:
# #1.Bar Chart: It displays a count of values present per columns ignoring missing values
# #2.Matrix: The nullity matrix chart lets us understand the distribution of data within the whole dataset in all columns at the same time which can help us understand the distribution of data better. It also displays sparkline which highlights rows with maximum and minimum nullity in a dataset.
# #3.Heatmap: The chart displays nullity correlation between columns of the dataset. It lets us understand how the missing value of one column is related to missing values in other columns.
# #4.Dendrogram: The dendrogram like heatmap groups columns based on nullity relation between them. It groups columns together where there is more nullity relation.
# 

# In[31]:


#msno.bar(dfDemographics, color="dodgerblue", sort="ascending", figsize=(10,5), fontsize=12);
msno.matrix(dfDemographics, figsize=(10,5), fontsize=12, color=(1, 0.38, 0.27));
#wont show as all missing values are handled
#msno.heatmap(dfDemographics, cmap="RdYlGn", figsize=(10,5), fontsize=12); 
#msno.dendrogram(dfDemographics, figsize=(10,5), fontsize=12);


# In[32]:


#msno.bar(dfMedicationHistory, color="dodgerblue", sort="ascending", figsize=(200,100), fontsize=100);
#msno.matrix(dfMedicationHistory1, figsize=(200,100), fontsize=100, color=(1, 0.38, 0.27));
msno.heatmap(dfMedicationHistory, cmap="RdYlGn", figsize=(200,100), fontsize=100); 
#msno.dendrogram(dfMedicationHistory1, figsize=(200,100), fontsize=100);


# In[33]:


dfMedicationHistory


# The values 1.5 x IQR (interquartile range) higher / smaller than Q3 / Q1 are called outliers. 
# IQR is the difference between Q3 and Q1 (IQR = Q3-Q1).
# One way to treat outliers is to make them equal to Q3 or Q1. By using pandas and numpy libraries,
# the below function does this task. Here. lower_upper_range function finds the range whose outside are outliers. Then with numpy clip function the values are clipped to the ranges.
# 
# 

# In[1]:


def number_of_outliers(df):
    
    df = df.select_dtypes(exclude = 'object')
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

def lower_upper_range(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range
  
for col in columns:  
    lowerbound,upperbound = lower_upper_range(df[col])
    df[col]=np.clip(df[col],a_min=lowerbound,a_max=upperbound)

