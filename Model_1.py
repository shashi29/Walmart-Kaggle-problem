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
# =============================================================================
# ea = sns.distplot(df_base['weekly_sales'],rug=True)
# eb = sns.lineplot(x=df_base['date'],y=df_base['weekly_sales'],data=df_base)
# #Sales look preety high during the duration of December
# ec = sns.lineplot(x=df_base['date'],y=df_base['temperature'],data=df_base)
# #Temperature is less during the high sales period
# #Let see how temperature and sales are co-rrelated
# ed = sns.scatterplot(y=df_base['weekly_sales'],x=df_base['temperature'],hue='isholiday' , data=df_base)
# #Higher sales are occured mostly in holidays 
# ee = sns.scatterplot(y=df_base['weekly_sales'],x=df_base['temperature'],hue='type' , data=df_base)
# ef = sns.scatterplot(y=df_base['weekly_sales'],x=df_base['temperature'],size='size' , data=df_base)
# eg = sns.scatterplot(y=df_base['weekly_sales'],x=df_base['temperature'],hue='isholiday' , data=df_base)
# 
# #Let use relpot
# eh = sns.relplot(y=df_base['weekly_sales'],x=df_base['temperature'],hue='isholiday' ,col='type', data=df_base)
# 
# #Let start exploring more on Fuel price
# #Need to work on the transformation for this also
# ef = sns.distplot(df_base['fuel_price'],rug=True)
# ee = sns.distplot(df_base['cpi'],rug=True)
# ej = sns.distplot(df_base['unemployment'],rug=True)
# ek = sns.distplot(df_base['temperature'],rug=True)
# tm = sns.countplot(df_base['year'])
# tm = sns.countplot(df_base['month'])
# tm = sns.countplot(df_base['day'])
# #Relplot
# r1 = sns.relplot(y='weekly_sales',x='unemployment',col='year',hue='type',data=df_base)
# 
# #Catplot
# c1 = sns.catplot(x='store',y='weekly_sales',col='year',data=df_base)
# c2 = sns.catplot(x='dept',y='weekly_sales',hue='year',data=df_base)
# c3 = sns.catplot(x='isholiday',y='weekly_sales',col='year',data=df_base)
# c4 = sns.catplot('weekly_sales',col='type',kind='count' ,data=df_base)
# 
# =============================================================================
#boxplot
b1 = sns.boxplot(x='year',y='weekly_sales',hue='type',data=df_base)
#Feature engineering for date
column_1 = df_base['date']

temp = pd.DataFrame({"year": column_1.dt.year,
              "month": column_1.dt.month,
              "day": column_1.dt.day,
              #"hour": column_1.dt.hour,
              "dayofyear": column_1.dt.dayofyear,
              "week": column_1.dt.week,
              "weekofyear": column_1.dt.weekofyear,
              #"dayofweek": column_1.dt.dayofweek,
              #"weekday": column_1.dt.weekday,
              "quarter": column_1.dt.quarter,
             })

df_base.reset_index(drop=True, inplace=True)
temp.reset_index(drop=True, inplace=True)
df_base = pd.concat([df_base,temp],axis=1)

df_base['weekly_sales'].min()
#Let deal with skewness problem of the target problem
df_base['log_sales'] = df_base['weekly_sales']+(4999)
#log is not working for removing the skewness of data

ea = sns.distplot(np.log1p(df_base['log_sales']),rug=True)

from scipy import stats
sales = stats.boxcox(df_base['log_sales'])
ea = sns.distplot(stats.boxcox(df_base['log_sales']),rug=True)
xt,_ = stats.boxcox(df_base['log_sales'])
fig = plt.figure()
ax2 = fig.add_subplot(212)
stats.probplot(xt, dist=stats.norm,plot=ax2)
plt.show()

ea = sns.distplot(xt,rug=True)

from numpy import mean
from numpy import std
print('mean=%.3f stdv=%.3f' % (mean(df_base['weekly_sales']), std(df_base['weekly_sales'])))
print('mean=%.3f stdv=%.3f' % (mean(xt), std(xt)))

plt.hist(df_base['weekly_sales'],bins='auto')
plt.show()
plt.hist(xt,bins='auto')
plt.show()


from statsmodels.graphics.gofplots import qqplot
qqplot(df_base['weekly_sales'], line='s')
plt.show()

qqplot(xt, line='s')
plt.show()

#Normality Test
from scipy.stats import shapiro
# normality test
#With this test the Transformed data also fail to be used
stat, p = shapiro(xt)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')

#df_base.drop(columns=['date'],axis=1,inplace=True)
#df_base = pd.get_dummies(df_base,columns=['type','isholiday'])

#Deal with skewness of data
#Sales = df_base['weekly_sales']
#df_base.drop(columns=['weekly_sales'],axis=1,inplace=True)

#Find Nan in each column
#No value is missing
df_base.isnull().sum()

#Total store : 45
#Total dept : 81
#Size Max 219622 Min 34875 count 421570




















'''

#Pre-processing for the test data
#Start training the regression model
test = pd.read_csv("test.csv",parse_dates = ['Date'])
test.columns = test.columns.str.lower()
test = pd.merge(test,stores,on='store',how='left')
test = pd.merge(test,features,on=['store','date','isholiday'],how='left')
test.isnull().sum()
#Values are missing in cpi : 38162
#Values are missing in unemployement : 38162



#Linear regression model
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df_base,Sales)
#Co-efficents for OST
#print('Coefficients: \n', reg.coef_)

'''