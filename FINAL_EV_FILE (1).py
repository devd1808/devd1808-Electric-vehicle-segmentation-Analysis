#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df = pd.read_csv('ElectricCarData_Clean.csv')


# In[8]:


df


# # Checking if Null Values are present 

# In[9]:


df.isna().sum()


# 

# In[10]:


df.info()


# In[11]:


df['PowerTrain'].unique()


# In[12]:


df['PlugType'].unique()


# In[13]:


df['BodyStyle'].unique()


# In[14]:


df['Segment'].unique()


# In[15]:


df['RapidCharge'].unique()


# In[16]:


df['FastCharge_KmH'].unique()


# # There were hypen present in FastCharge_KmH so we replaced it with mean of the attribute
# 

# In[17]:


df['FastCharge_KmH']=df['FastCharge_KmH'].str.replace('-','0')


# In[18]:


df['FastCharge_KmH']=df['FastCharge_KmH'].astype(float)


# In[19]:


df['FastCharge_KmH'].mean()


# In[20]:


df['FastCharge_KmH']=df['FastCharge_KmH'].replace(0, 434.56)


# In[21]:


df['FastCharge_KmH'].unique()


# # # # Droping Brand and Model name of Company Because we don't need these features for our analysis

# In[22]:


df=df.drop(['Brand','Model'],axis=1)


# In[23]:


df


# # EDA

# In[24]:


#Create a distribution plot for rating
sns.pairplot(df)
plt.show()


# In[25]:


#Plotting a pie chart
plt.figure(figsize=[9,7])
df['RapidCharge'].value_counts().plot.pie(autopct='%.0f%%')
plt.show()


# In[26]:


#Plotting a pie chart
plt.figure(figsize=[9,7])
df['Segment'].value_counts().plot.pie(autopct='%.0f%%')
plt.show()


# In[27]:


#Plotting a pie chart
plt.figure(figsize=[9,7])
df['PlugType'].value_counts().plot.pie(autopct='%.0f%%')
plt.show()


# In[28]:


sns.boxplot(x=df['TopSpeed_KmH'])
plt.show()
def convert(x):
     if x<160:
       return "Low"
     elif x>=160 and x<190:
       return "Medium"
     else:
       return "High"
converted_TopSpeed=df['TopSpeed_KmH'].apply(convert)
sns.countplot(x=converted_TopSpeed)
plt.show()


# In[29]:


sns.boxplot(x=df['Range_Km'])
plt.show()
def convert1(x):
     if x<250:
       return "Low"
     elif x>=250 and x<400:
       return "Medium"
     else:
       return "High"
converted_Range=df['Range_Km'].apply(convert1)
sns.countplot(converted_Range)
plt.show()


# In[30]:


sns.boxplot(x=df['Efficiency_WhKm'])
plt.show()
def convert2(x):
     if x<170:
       return "Low"
     elif x>=170 and x<200:
       return "Medium"
     else:
       return "High"
converted_Eff=df['Efficiency_WhKm'].apply(convert2)
sns.countplot(converted_Eff)
plt.show()


# In[31]:


sns.boxplot(x=df['AccelSec'])
plt.show()
def convert3(x):
     if x<6:
       return "Low"
     elif x>=6 and x<8:
       return "Medium"
     else:
       return "High"
converted_Acc=df['AccelSec'].apply(convert3)
sns.countplot(converted_Acc)
plt.show()


# In[32]:


df['PlugType'].value_counts()


# In[33]:


df['RapidCharge'].value_counts()


# # Since PlugType and RapidCharge are imbalanced attributes we drop these attributes

# In[34]:


from sklearn.preprocessing import OrdinalEncoder


# In[35]:



oe2=OrdinalEncoder(categories=[['AWD', 'RWD', 'FWD']])
oe4=OrdinalEncoder(categories=[['Sedan', 'Hatchback', 'Liftback', 'SUV', 'Pickup', 'MPV', 'Cabrio','SPV', 'Station']])
oe5=OrdinalEncoder(categories=[['D', 'C', 'B', 'F', 'A', 'E', 'N', 'S']])


# In[36]:



#df['PowerTrain']=oe2.fit_transform(df[['PowerTrain']])
#df['BodyStyle']=oe4.fit_transform(df[['BodyStyle']])
#df['Segment']=oe5.fit_transform(df[['Segment']])


# In[37]:


df = df.drop(columns=['RapidCharge','PlugType'])


# In[38]:


df = pd.get_dummies(df)
df


# In[ ]:





# In[39]:


df.info()


# In[40]:


df


# In[41]:


df=df.astype(float)


# In[42]:


df.info()


# In[43]:


cols_to_norm = ['AccelSec','TopSpeed_KmH','Range_Km','Efficiency_WhKm','FastCharge_KmH','PriceEuro']
df_prev=df.copy()
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# In[44]:


df


# In[45]:


from sklearn.cluster import KMeans


# In[46]:


wcss=[]
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(df)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,10)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# In[47]:


kmeanModel = KMeans(n_clusters=5)
kmeanModel.fit(df)


# In[48]:


identified_clusters = kmeanModel.fit_predict(df)
identified_clusters


# In[49]:


data_with_clusters = df_prev.copy()
data_with_clusters['Clusters'] = identified_clusters 


# In[50]:


data_with_clusters


# In[51]:


data_with_clusters['Clusters'].value_counts()


# In[52]:


Segment_wise_mean=data_with_clusters[['AccelSec','TopSpeed_KmH','Range_Km','Efficiency_WhKm','FastCharge_KmH','PriceEuro','Clusters','Seats']].groupby('Clusters').mean()


# In[53]:


Segment_wise_mean


# In[54]:


data_with_clusters[['AccelSec','TopSpeed_KmH','Range_Km','Efficiency_WhKm','FastCharge_KmH','PriceEuro','Clusters','Seats']].groupby('Clusters').min()


# In[55]:


data_with_clusters[['AccelSec','TopSpeed_KmH','Range_Km','Efficiency_WhKm','FastCharge_KmH','PriceEuro','Clusters','Seats']].groupby('Clusters').max()


# In[56]:


Segment_wise_sum=data_with_clusters.drop(columns=['AccelSec','TopSpeed_KmH','Range_Km','Efficiency_WhKm','FastCharge_KmH','PriceEuro'],axis=1).groupby('Clusters').sum()


# In[57]:


Segment_wise_sum


# In[ ]:





# In[ ]:




