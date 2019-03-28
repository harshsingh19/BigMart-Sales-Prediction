#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:40:18 2019

@author: harsh
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# Metrics for root mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from scipy.stats import mode

dftrain = pd.read_csv('Train.csv')
dftest  = pd.read_csv('Test.csv')
dftrain['source'] = 'train'
dftest['source'] = 'test'
dftest['Item_Outlet_Sales'] = 0
data=pd.concat([dftrain,dftest],ignore_index=True)
"""
-------Counting No of Null Values for each item id-------

No_null = []
for i in dftrain:
    for j in range(len(dftrain)):
        if not dftrain[i][j]:
            No_null.append(dftrain["Item_Identifier"][j])

counts = dict()
for word in No_null:
    if word in counts:
        counts[word] += 1
    else:
        counts[word] = 1
"""
data['Item_Outlet_Sales'].describe()


"""---------Skewness of the data------------
------------Ploating--------------
"""
sns.distplot(data['Item_Outlet_Sales'])
print("Shows Positive Skewness")
print('Skewness: %f' % data['Item_Outlet_Sales'].skew())

data.columns
"""Distgusing Categorical and Numerical Data"""
categorial_features = data.select_dtypes(include=[np.object])
numerical_features = data.select_dtypes(include=[np.number])
for x in categorial_features:
    print("\nfrequency of %s"%x)
    print(data[x].value_counts(dropna=False))
data['Outlet_Establishment_Year'].value_counts()
"---Counting Null Values in each coloum---"
data.apply(lambda x: sum(x.isnull()))
"---Counting Unique Values in each coloum---"
data.apply(lambda x : len(x.unique()))

"Ploating VS Graph using "
plt.figure(figsize = (10,9))

plt.subplot(311)
sns.barplot(x='Outlet_Size', y='Item_Outlet_Sales', data=data, palette="Set1",ci = 100)

plt.subplot(312)
sns.barplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', data=data, palette="Set1")

plt.subplot(313)
sns.barplot(x='Outlet_Type', y='Item_Outlet_Sales', data=data, palette="Set1")

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 1.5)

plt.show()


"""------ploating Graph --------"""
plt.figure(figsize = (14,9))

plt.subplot(211)
ax = sns.boxplot(x='Outlet_Identifier', y='Item_Outlet_Sales', data=data, palette="Set1")
ax.set_title("Outlet_Identifier vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)

plt.subplot(212)
ax = sns.boxplot(x='Item_Type', y='Item_Outlet_Sales', data=data, palette="Set1")
ax.set_title("Item_Type vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)

plt.subplots_adjust(hspace = 0.9, top = 0.9)
plt.setp(ax.get_xticklabels(), rotation=45)

plt.show()

print("Filling Missing values in Item Weight with mean")
data["Item_Weight"].fillna(value=dftrain["Item_Weight"].mean(),inplace=True)



"""--------Filling None Values in Categorical values--------------"""

outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x.astype('str')).mode[0]))
print ('Mode for each Outlet_Type:')
print (outlet_size_mode)
missing_values = data['Outlet_Size'].isnull() 
print ('\nOrignal #missing: %d'% sum(missing_values))
data.loc[missing_values,'Outlet_Size'] = data.loc[missing_values,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print (sum(data['Outlet_Size'].isnull()))



"""--------------Item Visiblity can not be zero so it needs to replaced with some value-------------"""

visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
missing_values = (data['Item_Visibility'] == 0)
print ('Number of 0 values initially: %d'%sum(missing_values))
data.loc[missing_values,'Item_Visibility'] = data.loc[missing_values,'Item_Identifier'].apply(lambda x: visibility_avg.at[x, 'Item_Visibility'])
print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))

"""--------Changing Type of Data------"""
data['Item_type_combined']=data['Item_Identifier'].apply(lambda x:x[0:2])
data['Item_type_combined']=data['Item_type_combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
data['Item_Fat_Content']=data['Item_Fat_Content'].map({'LF':'LowFat','reg':'Regular','low fat':'Low Fat','Regular':'Regular','Low Fat':'Low Fat'})
data['Item_Fat_Content'].value_counts().sum()


"""Establishment Year is Import Feature so we are taking some analysis of this coloum"""

data.index = data['Outlet_Establishment_Year']
data.index
df = data.loc[:,['Item_Outlet_Sales']]
data.groupby('Outlet_Establishment_Year')['Item_Outlet_Sales'].mean().plot.bar()
data['Outlet_Years'] = 2009 - data['Outlet_Establishment_Year']


plt.figure(figsize = (12,6))
ax = sns.boxplot(x = 'Outlet_Years', y = 'Item_Outlet_Sales', data = data)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
ax.set_title('Outlet years vs Item_Outlet_Sales')
ax.set_xlabel('', fontsize = 15)
ax.set_ylabel('Item_Outlet_Sales', fontsize = 15)
plt.show()


temp_data = data.loc[data['Outlet_Establishment_Year'] == 1998]
temp_data['Outlet_Type'].value_counts()

"""Label Encoder for converting text data into Numeric form as Machine does not understand Numeric way"""

le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_type_combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])

data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_type_combined','Outlet'])
data['Item_Identifier'] = le.fit_transform(data['Item_Identifier'])
    
    
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source','Outlet_Identifier'],axis=1,inplace=True)
train.drop(['source','Outlet_Identifier'],axis=1,inplace=True)
corr=data.corr()

y = train['Item_Outlet_Sales']
train.drop('Item_Outlet_Sales',axis=1,inplace=True)
"""------Using Multiple linear Regression To Predict the values"""
lr = LinearRegression();
lr.fit(train, y)
y_pred = lr.predict(test)
print(sqrt(mean_squared_error(np.log(y), np.log(y_pred))))