#!/usr/bin/env python
# coding: utf-8

# # 员工离职预测
# ##### 参赛地址：http://www.dcjingsai.com/common/cmpt/%E5%91%98%E5%B7%A5%E7%A6%BB%E8%81%8C%E9%A2%84%E6%B5%8B%E8%AE%AD%E7%BB%83%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html

# In[316]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[317]:


#取数
train = pd.read_csv('/Users/cy_ariel/Desktop/员工离职预测训练赛/pfm_train.csv')
test = pd.read_csv('/Users/cy_ariel/Desktop/员工离职预测训练赛/pfm_test.csv')


# In[318]:


train.info()
print('-----------')
test.info()


# In[319]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(corrmat)


# In[320]:


k=29
corrmat.nlargest(k,'Attrition')['Attrition']


# In[321]:


train_df = train.drop(['EmployeeNumber','Over18','MonthlyIncome','YearsWithCurrManager','YearsAtCompany','TotalWorkingYears','PercentSalaryHike'],axis=1)
test_df = test.drop(['EmployeeNumber','Over18','MonthlyIncome','YearsWithCurrManager','YearsAtCompany','TotalWorkingYears','PercentSalaryHike'],axis=1)


# In[322]:


train_df.info()
print('------------')
test_df.info()


# In[323]:


train_df[['OverTime','Attrition']].groupby('OverTime',as_index=False).mean().sort_values(by='Attrition')


# In[324]:


dum1_train = pd.get_dummies(train_df['OverTime'])
dum1_train.set_axis(['overtime_No','overtime_Yes'],axis=1)
train_df = pd.concat([train_df,dum1_train],axis=1)
train_df = train_df.drop(['OverTime','overtime_No'],axis=1)
dum1_test = pd.get_dummies(test_df['OverTime'])
dum1_test.set_axis(['overtime_No','overtime_Yes'],axis=1)
test_df = pd.concat([test_df,dum1_test],axis=1)
test_df = test_df.drop(['OverTime','overtime_No'],axis=1)


# In[325]:


train_df[['BusinessTravel','Attrition']].groupby('BusinessTravel',as_index=False).mean().sort_values(by='Attrition')


# In[326]:


dum2_train = pd.get_dummies(train_df['BusinessTravel'])
train_df = pd.concat([train_df,dum2_train],axis=1)
train_df = train_df.drop(['BusinessTravel','Non-Travel'],axis=1)
dum2_test = pd.get_dummies(test_df['BusinessTravel'])
test_df = pd.concat([test_df,dum2_test],axis=1)
test_df = test_df.drop(['BusinessTravel','Non-Travel'],axis=1)


# In[327]:


train_df[['MaritalStatus','Attrition']].groupby('MaritalStatus',as_index=False).mean().sort_values(by='Attrition')


# In[328]:


dum3_train = pd.get_dummies(train_df['MaritalStatus'])
train_df = pd.concat([train_df,dum3_train],axis=1)
train_df = train_df.drop(['MaritalStatus','Divorced'],axis=1)
dum3_test = pd.get_dummies(test_df['MaritalStatus'])
test_df = pd.concat([test_df,dum3_test],axis=1)
test_df = test_df.drop(['MaritalStatus','Divorced'],axis=1)


# In[329]:


train_df[['JobRole','Attrition']].groupby('JobRole',as_index=False).mean().sort_values(by='Attrition')


# In[330]:


dum4_train = pd.get_dummies(train_df['JobRole'])
dum4_train.set_axis(['jobrole_Research','jobrole_Healthcare','jobrole_Manager','jobrole_Manufacturing','jobrole_Executive','jobrole_Scientist','jobrole_Laboratory','jobrole_Human','jobrole_Representative'],axis=1)
train_df = pd.concat([train_df,dum4_train],axis=1)
train_df = train_df.drop(['JobRole','jobrole_Research'],axis=1)
dum4_test = pd.get_dummies(test_df['JobRole'])
dum4_test.set_axis(['jobrole_Research','jobrole_Healthcare','jobrole_Manager','jobrole_Manufacturing','jobrole_Executive','jobrole_Scientist','jobrole_Laboratory','jobrole_Human','jobrole_Representative'],axis=1)
test_df = pd.concat([test_df,dum4_test],axis=1)
test_df = test_df.drop(['JobRole','jobrole_Research'],axis=1)


# In[331]:


train_df[['Gender','Attrition']].groupby('Gender',as_index=False).mean().sort_values(by='Attrition')


# In[332]:


dum5_train = pd.get_dummies(train_df['Gender'])
train_df = pd.concat([train_df,dum5_train],axis=1)
train_df = train_df.drop(['Gender','Female'],axis=1)
dum5_test = pd.get_dummies(test_df['Gender'])
test_df = pd.concat([test_df,dum5_test],axis=1)
test_df = test_df.drop(['Gender','Female'],axis=1)


# In[333]:


train_df = train_df.drop(['EducationField'],axis=1)
test_df = test_df.drop(['EducationField'],axis=1)


# In[334]:


train_df = train_df.drop(['Department'],axis=1)
test_df = test_df.drop(['Department'],axis=1)


# In[335]:


X_train = train_df.loc[:,train_df.columns!='Attrition']
Y_train = train_df.loc[:,train_df.columns=='Attrition']


# In[336]:


train_df.info()


# ## LR建模

# In[337]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[338]:


X_train_1 = X_train
Y_train_1 = Y_train


# In[342]:


test_1 = test_df


# In[340]:


lr.fit(X_train,Y_train)


# In[343]:


y_pre = lr.predict(test_1)


# In[344]:


df = pd.DataFrame(y_pre)


# In[345]:


df.set_axis(['result'],axis=1)


# In[346]:


df.to_csv('/Users/cy_ariel/Desktop/员工离职预测训练赛/result.csv')


# ### 随机森林建模

# In[347]:


from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

random_forest = RandomForestClassifier()

random_forest.fit(X_train_1, Y_train_1)

Y_pred = random_forest.predict(test_1)

random_forest.score(X_train_1, Y_train_1)


# In[348]:


y_pre = pd.DataFrame(Y_pred)


# In[349]:


y_pre.set_axis(['result'],axis=1)


# In[350]:


y_pre.to_csv('/Users/cy_ariel/Desktop/员工离职预测训练赛/result.csv')

