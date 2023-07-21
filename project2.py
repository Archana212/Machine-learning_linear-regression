# importing required libraries
import numpy as np
import pandas as pd
import pylab as pl
import statsmodels.api as sm

#importing the dataset
df=pd.read_csv("binary.csv")
df.head()
#rename column names
df.columns=["admit","gre","gpa","prestige"]
df.head()

df.shape

df.describe()

#frequency according to prestige column
pd.crosstab(df['admit'],['prestige'],rownames=['admit'])

df.hist()
pl.show()

#dummy columns
dummy_ranks=pd.get_dummies(df['prestige'],prefix='prestige')
dummy_ranks.head()

#create the dataframe for regression
cols_to_keep=['admit','gre','gpa']
data=df[cols_to_keep].join(dummy_ranks.loc[:,'pestige_2':])
data.head()

#manually add the intercept 
data['intercept']=1.0
data.head()

#performing regression
train_cols=data.columns[1:]

logit=sm.Logit(data['admit'],data[train_cols])

result=logit.fit()

#now predeit by giving some values

abc = result.predict([660,3.67,0,1,0,1.0])
print(abc)

result.summary()














