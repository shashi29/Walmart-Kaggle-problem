import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

train = pd.read_csv('train.csv', parse_dates=['Date'])
train.columns = train.columns.str.lower()
features = pd.read_csv('features.csv', parse_dates=['Date'])
features.columns = features.columns.str.lower()
features[['markdown1', 'markdown2', 'markdown3','markdown4', 'markdown5']] = features[['markdown1', 'markdown2', 'markdown3','markdown4', 'markdown5']].fillna(0)
stores = pd.read_csv('stores.csv')
stores.columns = stores.columns.str.lower()
df_base = pd.merge(train,stores,on='store',how='left')
df_base = pd.merge(df_base,features,on=['store','date','isholiday'],how='left')

#Looking at this it clears the weekly sales is skewed on left side
#+ve skewed distribution
ez = sns.distplot(df_base['weekly_sales'],rug=True)

#Feature engineering for date
column_1 = df_base['date']

temp = pd.DataFrame({"year": column_1.dt.year,
              "month": column_1.dt.month,
              "day": column_1.dt.day,
              "hour": column_1.dt.hour,
              "dayofyear": column_1.dt.dayofyear,
              "week": column_1.dt.week,
              "weekofyear": column_1.dt.weekofyear,
              "dayofweek": column_1.dt.dayofweek,
              "weekday": column_1.dt.weekday,
              "quarter": column_1.dt.quarter,
             })

df_base.reset_index(drop=True, inplace=True)
temp.reset_index(drop=True, inplace=True)
df_base = pd.concat([df_base,temp],axis=1)

df_base.drop(columns=['date'],axis=1,inplace=True)
df_base = pd.get_dummies(df_base,columns=['type','isholiday'])

#Deal with skewness of data
Sales = df_base['weekly_sales']
df_base.drop(columns=['weekly_sales'],axis=1,inplace=True)

#Find Nan in each column
#No value is missing
df_base.isnull().sum()

#Pre-processing for the test data
#Start training the regression model
test = pd.read_csv("test.csv",parse_dates = ['Date'])
test.columns = test.columns.str.lower()
test = pd.merge(test,stores,on='store',how='left')
test = pd.merge(test,features,on=['store','date','isholiday'],how='left')
test.isnull().sum()
#Values are missing in cpi
#Values are missing in unemployement

#Linear regression model
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df_base,Sales)
#Co-efficents for OST
#print('Coefficients: \n', reg.coef_)
