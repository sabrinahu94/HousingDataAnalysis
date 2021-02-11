#!/usr/bin/env python
# coding: utf-8

# ## Project 2 - Data Cleansing Practice on Zillow Data

# In this week, we’ll practice how to do regular cleansing in Python with a real-world dataset – Zillow dataset. This Zillow dataset contains abundant missing data and will provide you a good environment to practice your skills on data cleaning.

# This step-by-step project will illustrate you various ways to impute missing values.

# In[1]:


# Start with importing essentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### 1. Read the train set and property set of Zillow dataset, and name them as train and properties.

# In[2]:


properties = pd.read_csv('input/properties_2016.csv')
train = pd.read_csv("input/train_2016_v2.csv")


# #### 2. Merge train and properties to one dataframe on parcelid and call it as df_train. Drop the column of 'parcelid' and 'transactiondate'. Check the first 5 rows to see how this merged dataset looks like.

# In[3]:


df_train = train.merge(properties, how='left', on='parcelid')
df_train.drop(['parcelid', 'transactiondate'], axis=1,inplace=True)


# In[4]:


df_train.head(5)


# #### 3.  (a) Generate a dataframe called missing_df from df_train, in which there are two columns, one is the column names of our features, the other column is the missing_count (the number of missing values) of that feature. The table should be ordered by missing_count decendingly.  

# In[ ]:





# In[137]:


missing_df = df_train.isnull().sum().reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')


# In[141]:


# missing_df = pd.DataFrame(df_train.isnull().sum().reset_index().values, columns=['column_name', 'missing_count'])
# missing_df = missing_df[missing_df['missing_count']>0].sort_values(by='missing_count')
# missing_df.head(7)


# In[148]:


# missing_df.plot(x='column_name', y='missing_count', kind='barh', figsize=(14,14), logx=False )
# plt.show()


# #### 3.(b) Draw a horizontal bar plot to visualize it. Following is an example to show how this figure may look like:

# In[145]:


ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# #### 4. Generate the correlation matrix for all the numerical features, and plot it by using heatmap or related visualization methods. 

# In[96]:


## Classify the features into categorical and numerical fearures
catcols = ['airconditioningtypeid','architecturalstyletypeid','buildingqualitytypeid','buildingclasstypeid','decktypeid','fips','hashottuborspa','heatingorsystemtypeid','pooltypeid10','pooltypeid2','pooltypeid7','propertycountylandusecode','propertylandusetypeid','propertyzoningdesc','rawcensustractandblock','regionidcity','regionidcounty','regionidneighborhood','regionidzip','storytypeid','typeconstructiontypeid','yearbuilt','taxdelinquencyflag']
numcols = [x for x in df_train.columns if x not in catcols]


# In[97]:


numcols


# In[98]:


plt.figure(figsize = (14,12))
sns.heatmap(data=df_train[numcols].corr())
plt.show()


# #### 5. From the results from Step 4, please list those features having a strong correlation. Generate a list called dropcols, and put those redundent variables into it.

# In[100]:


dropcols = []
## 'calculatedfinishedsquarefeet' 'finishedsquarefeet12' 'finishedsquarefeet13' 'finishedsquarefeet15' 
## 'finishedsquarefeet6' has strong correlations, but 'calculatedfinishedsquarefeet' doesn't have missing value, so 
## we keep it and delete the rest
dropcols.append('finishedsquarefeet12')
dropcols.append('finishedsquarefeet13')
dropcols.append('finishedsquarefeet15')
dropcols.append('finishedsquarefeet6')


# In[101]:


## finishedsquarefeet50 and finishedfloor1squarefeet are the exactly the same information according to the dictionary descriptions, 
## lets remove finishedsquarefeet50 considering it has more missing values
dropcols.append('finishedsquarefeet50')


# In[102]:


##'bathroomcnt' and 'calculatedbathnbr' and 'fullbathcnt' have high relationship as well.
## We keep'bathroomcnt' as has no missing values
dropcols.append('calculatedbathnbr')
dropcols.append('fullbathcnt')


# #### 6. Some variables where it is NA can be considered as the object does not exist. Such as 'hashottuborspa', if it is NA, we can assume the house doesn't contain the hot tub or spa. So we need to fix this.

# (a) Fix the hashottuborspa variable, fill the na part as None.

# In[103]:


index = df_train.hashottuborspa.isnull()
df_train.loc[index,'hashottuborspa'] = "N/A"


# (b) Assume if the pooltype id and its related features is null then pool/hottub doesnt exist.

# In[104]:


index = df_train.pooltypeid2.isnull()
df_train.loc[index,'pooltypeid2'] = 0


# In[105]:


index = df_train.pooltypeid7.isnull()
df_train.loc[index,'pooltypeid7'] = 0


# In[106]:


index = df_train.pooltypeid10.isnull()
df_train.loc[index,'pooltypeid10'] = 0


# In[107]:


index = df_train.poolcnt.isnull()
df_train.loc[index,'poolcnt'] = 0


# (c) Tax deliquency flag - assume if it is null then doesn't exist

# In[108]:


index = df_train.taxdelinquencyflag.isnull()
df_train.loc[index,'taxdelinquencyflag'] = "None"


# (d) If Null in garage count (garagecarcnt) it means there are no garages, and no garage means the size (garagetotalsqft) is 0 by default

# In[109]:


index = df_train.garagecarcnt.isnull()
df_train.loc[index,'garagecarcnt'] = 0


# In[110]:


index = df_train.garagetotalsqft.isnull()
df_train.loc[index,'garagetotalsqft'] = 0


# #### 7. There are more missing values in the 'poolsizesum' than in 'poolcnt'. Fill in median values for poolsizesum where pool count is >0 and missing.

# In[112]:


#Fill in those properties that have a pool with median pool value
poolsizesum_median = df_train.loc[df_train['poolcnt'] > 0, 'poolsizesum'].median()
df_train.loc[(df_train['poolcnt'] > 0) & (df_train['poolsizesum'].isnull()), 'poolsizesum'] = poolsizesum_median

#If it doesn't have a pool then poolsizesum is 0 by default
df_train.loc[(df_train['poolcnt'] == 0), 'poolsizesum'] = 0


# #### 8. The number of missing value of 'fireplaceflag' is more than the 'fireplacecnt'. So we need to mark the missing 'fireplace flag' as Yes when fireplacecnt>0, then the rest of 'fireplaceflag' should be marked as No. Then for the missing part in fireplacecnt, we can consider the number of fire place is 0.

# In[113]:


df_train['fireplaceflag']= "No"
df_train.loc[df_train['fireplacecnt']>0,'fireplaceflag']= "Yes"

index = df_train.fireplacecnt.isnull()
df_train.loc[index,'fireplacecnt'] = 0


# #### 9. Fill some features with the most common value for those variables where this might be a sensible approach:

# (a) AC Type (airconditioningtypeid)- Mostly 1's, which corresponds to central AC. It is reasonable to assume most other properties where this feature is missing are similar.

# In[43]:


index = df_train.airconditioningtypeid.isnull()
df_train.loc[index,'airconditioningtypeid'] = 1


# (b) heating or system (heatingorsystemtypeid)- Mostly 2, which corresponds to central heating so seems reasonable to assume most other properties have central heating.

# In[44]:


index = df_train.heatingorsystemtypeid.isnull()
df_train.loc[index,'heatingorsystemtypeid'] = 2


# #### 10. If the features where missing proportion is too much, we can directly delete them. Here we set 97% as our threshold (This is subjective) and add them into the dropcols. Then drop those features in dropcols from the full table.

# In[149]:


missingvalues_prop = (df_train.isnull().sum()/len(df_train)).reset_index()  
missingvalues_prop.columns = ['field','proportion']
missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)
print(missingvalues_prop)
missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.97].field.tolist()
dropcols = dropcols + missingvaluescols
df_train = df_train.drop(dropcols, axis=1)

