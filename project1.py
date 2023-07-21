import pandas as pd
import matplotlib.pyplot as plt
#read data into a DataFrame
data=pd.read_csv('advertising.csv')
data.head()
#print the shape of the dataFrame
data.shape

fig,axs=plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[2])
#create X and y
feature_cols=['TV']
X=data[feature_cols]
y=data.Sales

from  sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X,y)
#print intrcept and coefficient
print(lm.intercept_)
print(lm.coef_)

6.97482+0.055464*50

#lm.predict(X_new)

X_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

preds=lm.predict(X_new)
preds

data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(X_new,preds,c='red',linewidth=2)

import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales~TV',data=data).fit()
lm.conf_int()

lm.pvalues
#print the r squared value for the model
lm.rsquared

feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
y=data.Sales

lm=LinearRegression()
lm.fit(X,y)

print(lm.intercept_)
print(lm.coef_)

lm=smf.ols(formula='Sales~TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()

#only include tv and radio in model
lm=smf.ols(formula='Sales~TV+Radio',data=data).fit()
lm.rsquared

#include catogerical variables
import numpy as np

#set a seed for reproductaility
np.random.seed(12345)

#create a Series of booleans in which roughly half are True
nums=np.random.rand(len(data))
mask_large=nums>0.5

#intially set size to small,then change roughly half to be large
data['Size']='small'
data.loc[mask_large,'Size']='large'
data.head()

#create a new series called IsLarge
data['IsLarge']=data.Size.map({'small':0,'large':1})
data.head()

#create X and y
feature_cols=['TV','Radio','Newspaper','IsLarge']
X=data[feature_cols]
y=data.Sales

#instantiate,fit
lm=LinearRegression()
lm.fit(X,y)

#print coefficients
print(feature_cols,lm.coef_)

#handling categorial predictors with more than 2 categories

#set a seed for reproductability
np.random.seed(123456)

#assign roughly one third of observations to each group
nums=np.random.rand(len(data))
mask_suburban=(nums>0.33)&(nums<0.66)
mask_urban=nums>0.66
data['Area']='rural'
data.loc[mask_suburban,'Area']='suburban'
data.loc[mask_urban,'Area']='urban'
data.head()

#create 3 dummy varibles using get_dummies,then exclude the first
area_dummies=pd.get_dummies(data.Area,prefix='Area').iloc[:,1:]
area_dummies

#conctenate the dummy variable columns onto the original dataframe
data=pd.concat([data,area_dummies],axis=1)
data.head()

#create X and y
feature_cols=['TV','Radio','Newspaper','IsLarge','Area_suburban','Area_urban']
X=data[feature_cols]
y=data.Sales

#instantiate,fit
lm=LinearRegression()
lm.fit(X,y)

#print coefficients
print(feature_cols,lm.coef_)