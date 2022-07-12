#!/usr/bin/env python
# coding: utf-8

# ******************PANDAS PROFILING LIBRARY**********************

# In[ ]:


#!pip install MarkupSafe 
#!pip install markupsafe==2.0.1 
#!pip install --user --upgrade aws-sam-cli
#!pip install --user --upgrade aws-sam-cli


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[8]:


get_ipython().system('pip install --user pandas-profiling')
from pandas_profiling import ProfileReport


# In[4]:


#df=pd.read_excel("Diabetes-Modified .xlsx")
dfDemographics=pd.read_excel("Diabetes-Modified .xlsx","Demographics in V2 and V8")
#dfMedicationHistory=pd.read_excel("Diabetes-Modified .xlsx","Medical and Medication History ")
#dfLabTests=pd.read_excel("Diabetes-Modified .xlsx","Lab Tests in V2 and V8")
#dfMRI=pd.read_excel("Diabetes-Modified .xlsx","MRI Tests in V2 and V8")
#dfCognitive=pd.read_excel("Diabetes-Modified .xlsx","Cognitive Tests")


# In[5]:


dfDemographics.head(10)


# In[9]:


profile = ProfileReport(dfDemographics)
#profile = ProfileReport(df.sample(n=100),title="Pandas Profiling Report", minimal=True,explorative=True) #smaller chunk of larger dataset
profile.to_file(output_file ='projectdiabetesoutput.html')


# In[ ]:


profile.to_widgets()
profile.to_notebook_iframe()


# *****************************DTALE LIBRARY************************

# In[2]:


get_ipython().system('pip install dtale')


# In[6]:


import dtale
dtale.show(dfDemographics)


# ******************************SWEETVIZ LIBRARY****************************

# In[10]:


get_ipython().system('pip install sweetviz')


# In[13]:


import sweetviz as sv
report=sv.analyze(dfDemographics)
report.show_html('dtaleoutputreport.html')


# In[ ]:


import pandas as pd
import numpy as np
pip install pandas-profiling
demo = pd.read_excel('Diabetes.xlsx', index_col=None, usecols="A,A:J")
#profiling for demo data
prof_demo = ProfileReport(demo)
#preprocessing of demographics data
demo['Diabetes Duration'] = demo['Diabetes Duration'].replace(np.nan, 0)
demo['HTN or not'] = demo['HTN or not'].replace('ntn', 0).replace('HTN', 1)
demo['Group'] = demo['Group'].replace('CONTROL', 0).replace('DM', 1)
demo['DM, Non-DM, STROKE'] = demo['DM, Non-DM, STROKE'].replace('Non-DM', 0).replace('DM', 1)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
demo['Race'] = le.fit_transform(demo['Race'])
demo['HTN or not'] = demo['HTN or not'].fillna(demo['HTN or not'].mode()[0])

# cognitive data
cog_test = pd.read_excel('Diabetes.xlsx', index_col=None, usecols="AVF,AVF:AXQ")
prof_cog_test = ProfileReport(cog_test)

