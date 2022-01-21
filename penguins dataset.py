#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv("G:\Penguindata.csv")
df.head(3)


# ## Data Exploration and Cleaning

# In[5]:


df.shape


# In[6]:


df.duplicated().sum()


# In[7]:


df.info()


# #### dropping the comments column as it contains so many missing values and it also won't really be needed for visualising the data 

# In[8]:


df.drop(["Comments"],axis = 1,inplace = True)
df.head(2)


# In[9]:


df.shape


# In[10]:


def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        table = pd.concat([mis_val, mis_val_percent], axis=1)

        col_names = table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        col_names= col_names[col_names.iloc[:,1] != 0].sort_values('% of Total Values',
                                                                   ascending=False).round(1)
        
        print ("Dataframe has " + str(df.shape[1]) + " columns,"      
            "And there exist " + str(col_names.shape[0]) +
              " columns missing value(s)")
        
        return col_names


# In[11]:


summary=missing_values_table(df)
summary


# ## Handling Missing Values 

# In[12]:


df=df.dropna(subset=["Culmen Length (mm)","Culmen Depth (mm)","Flipper Length (mm)",
                     "Body Mass (g)","Sex","Delta 15 N (o/oo)","Delta 13 C (o/oo)"])
df.shape


# In[13]:


df.isnull().sum()


# # Data Visualization

# In[14]:


df['Species'].value_counts()


# In[15]:


x=df['Species'].value_counts().index
print(x)
print()
y=df['Species'].value_counts().values.tolist()
print(y)


# # Pie Chart

# In[16]:


plt.figure(figsize=(8,8))
pal = sns.color_palette("Paired")
#explode needs a length to function so making a for loop makes it more dynamic
plt.pie(y, labels=x, colors=pal,autopct='%1.1f%%',
        explode=[0.03 for i in df['Species'].value_counts().index])
plt.show()


# # Line Histogram

# ### hist returns a tuple of 3 values (n, bins, patches) :n= array or list of arrays(The values of the histogram bin) / bins = bin edges / patches = list or list of lists (list of individual patches used to create the histogram)

# In[17]:


import scipy.stats as stats
#Kernel density estimation is a way to estimate 
#the probability density function (PDF) of a random variable
density = stats.gaussian_kde(df["Body Mass (g)"])
plt.style.use("ggplot")
plt.figure(figsize=(10,5))
n, bins, patches = plt.hist(df["Body Mass (g)"],
                            bins=20 ,
                            color="lightseagreen",
                            density=True,
                            alpha =0.6,
                            edgecolor="white")
print("histogram edges")
print(bins)
plt.xlabel('Body Mass')
plt.ylabel('Probability')
plt.title("Line Histogram")
plt.plot(bins,density(bins))
plt.show()


# # Column Histogram

# In[18]:


plt.figure(figsize=(10,5))
plt.hist(df["Culmen Length (mm)"],color='yellowgreen',
         bins=20,alpha =0.7,edgecolor="white")
plt.show()


# In[19]:


df["Island"].value_counts()


# #### notice how there is a wrong record 

# In[20]:


df["Sex"].value_counts()


# In[21]:


df.drop(df.index[df['Sex'] == "."], inplace = True)


# # Bubble Chart

# In[22]:


plt.figure(figsize=(10, 5))
sns.scatterplot(x="Culmen Length (mm)", 
                y="Culmen Depth (mm)",
                size="Body Mass (g)",
                #creates the range
                sizes=(20,500),
                alpha=0.5,
                #to color the bubble chart by the fourth variable 
                hue="Sex",
                data=df,
                palette="husl")

plt.legend(bbox_to_anchor=(1.0, 0.8))
plt.xlabel("Culmen Length (mm)")
plt.ylabel("Culmen Depth (mm)")
plt.title("Bubble Plot")
plt.show()


# # Scatter Chart 

# In[23]:


plt.figure(figsize=(10,5))
sns.scatterplot(x="Culmen Length (mm)", 
                y="Culmen Depth (mm)", 
                hue="Species",
                data=df,
                palette="mako",
                alpha=0.8)
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)
plt.xlabel("Culmen Length (mm)")
plt.ylabel("Culmen Depth (mm)")
plt.show()


# # Tree chart

# In[24]:


import squarify
size=list(df['Species'].value_counts())
label=["Adelie","Gentoo","Chinstrap"]
squarify.plot(sizes=size, label=label)
plt.show()


# # subplots

# In[25]:


fig, axes = plt.subplots(2, 1,figsize=(8,8))

axes[0].hist('Culmen Length (mm)',bins=20,color='thistle',
             edgecolor='white',data=df)
axes[0].set_xlabel('Culmen Length (mm)')
axes[0].set_ylabel('counts')

axes[1].hist('Culmen Depth (mm)',bins=20,color='lightsteelblue',
             edgecolor='white',data=df)
axes[1].set_xlabel('Culmen Depth (mm)')
axes[1].set_ylabel('counts')

plt.show()


# # box plot

# In[26]:


df1 = df[['Culmen Length (mm)','Culmen Depth (mm)','Flipper Length (mm)']]
plt.figure(figsize = (10, 5))
df1.boxplot(color ="coral")
plt.show()

