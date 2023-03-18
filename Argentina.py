#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import LinearRegression


# In[6]:


df = pd.read_csv('argentina_cars.csv')


# In[7]:


df.head()


# In[8]:


#Need to firstly get all the data into dollars, as we have dollars and pesos 

conversion_rate = 0.0053

df['money'] = np.where(df['currency'] == 'pesos', df['money'] * conversion_rate, df['money'])
df['currency'] = np.where(df['currency'] == 'pesos', 'dólares', df['currency'])


# In[9]:


df.head()


# In[10]:


#Accents are annoying so I'll change Automática to Automatic and Dolares to Dollars 
df['currency'] = np.where(df['currency'] == 'dólares', 'dollars', df['currency'])
df['gear'] = np.where(df['gear'] == 'Automática', 'Automatic', df['gear'])


# In[11]:


df.head()


# In[12]:


#check where the nulls are 
null_counts = df.isnull().sum()
print(null_counts)


# In[13]:


df.dropna(inplace=True)


# In[14]:


df.head()


# In[15]:


#Start of the EDA - How many different brands are there and visualise this 
unique_brands = df['brand'].nunique()
print(f'Number of Unique Brands: {unique_brands}')


# In[16]:


brand_counts = df['brand'].value_counts()
brand_counts.plot(kind='bar')

plt.title("Brand Counts")
plt.xlabel("Brand")
plt.ylabel("Count")

plt.show()


# In[17]:


fig, axs = plt.subplots(ncols=2,figsize=(15,4))

sns.distplot(df["money"],color="green", ax=axs[0])
sns.distplot(df["money"], rug=True,ax=axs[0], rug_kws={"color": "black"},
                  kde_kws={"color": "black", "lw": 4, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "black"})

sns.boxplot(x="money",
             color="green",
            data=df,ax=axs[1])

fig.suptitle('Distribution of Car Price in the Dataset', y=1)

plt.grid(False)
plt.show();


# In[18]:


# Need to get rid of outliers 
money_mean = df['money'].mean()
money_std = df['money'].std()

df['money_zscore'] = (df['money'] - money_mean) / money_std

df = df[(df['money_zscore'] > -3) & (df['money_zscore'] < 3)]

df.head


# In[34]:


sns.set_palette('bright')
sns.set_theme(palette='deep')
sns.catplot(x='fuel_type', y='money', data=df)
sns.boxplot(x='fuel_type', y='money', data=df)
plt.show()


# In[29]:


fig = plt.figure(figsize=(30,8))
sns.violinplot(x='year', y='money', data=df)
plt.show()
plt.suptitle('Car Price By Year of Make')
plt.show()


# In[44]:


df['brand'] = pd.Categorical(df['brand'])
dumies = pd.get_dummies(df['brand'], prefix = 'brand')
df = pd.concat([df,dumies], axis = 1)


# In[47]:


df['fuel_type'] = pd.Categorical(df['fuel_type'])
dumies = pd.get_dummies(df['fuel_type'], prefix = 'fuel_type')
df = pd.concat([df,dumies], axis = 1)


# In[50]:


df['gear'] = pd.Categorical(df['gear'])
dumies = pd.get_dummies(df['gear'], prefix = 'gear')
df = pd.concat([df,dumies], axis = 1)


# In[51]:


df['body_type'] = pd.Categorical(df['body_type'])
dumies = pd.get_dummies(df['body_type'], prefix = 'body_type')
df = pd.concat([df,dumies], axis = 1)


# In[64]:


df['motor'] = pd.to_numeric(df['motor'], errors='coerce')
df = df.dropna()


# In[65]:


df_model = df.drop(['color','currency', 'money_zscore', 'model', 'brand', 'gear', 'body_type', 'fuel_type'], axis=1)


# In[66]:


df_model.isna().sum()


# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X=df_model.drop('money', axis=1)
y =df_model['money']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[71]:


model= LinearRegression()


# In[72]:


model.fit(X_train, y_train)


# In[75]:


c = model.intercept_
print(c)


# In[76]:


m = model.coef_
m


# In[79]:


y_pred_train = model.predict(X_train)


# In[80]:


plt.scatter(y_train, y_pred_train)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')


# In[73]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)
print(f"Mean squared error: {mse}")
print(f"R-squared coefficient: {r2}")


# In[82]:


from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits = 6, shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)


# In[83]:


print('Cross-validation scores:', cv_results)
print('Mean cross-validation score:', cv_results.mean())


# In[ ]:




