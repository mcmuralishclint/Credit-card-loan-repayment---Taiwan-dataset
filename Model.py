# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 13:31:52 2018

@author: muralish
"""

"""Limit_balance - Amount of given credit
Sex - 1:Male   2:Female
Education - 1:grad school 2:university 3:High school 4:others
marriage - 1:Married 2:Single 3:Others
Pay_0 to Pay_6 - pay_0 is the repayment history in sep, pay_1 is aug, pay_2 is july
    repayment status: -1 is pay duly, 1 is delay for a month, 2 is delay for 2 months"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('UCI_Credit_Card.csv',index_col="ID")

data['grad_school']=(data['EDUCATION']==1).astype('int')
data['university']=(data['EDUCATION']==2).astype('int')
data['high_school']=(data['EDUCATION']==3).astype('int')
data.drop('EDUCATION',axis=1,inplace=True)

data['male']=(data['SEX']==1).astype('int')
data.drop('SEX',axis=1,inplace=True)



data['married'] = (data['MARRIAGE']==1).astype('int')
data['single'] = (data['MARRIAGE']==2).astype('int')
data.drop('MARRIAGE',axis=1,inplace=True)

pay_features = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_5']
for p in pay_features:
    data.loc[data[p]<=0,p]=0
    
data.rename(columns={'default.payment.next.month':'default'},inplace=True)



#build models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.preprocessing import RobustScaler

target = 'default'
x = data.drop(target,axis=1)
robustscaler = RobustScaler()
x = robustscaler.fit_transform(x)
y= data[target]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 42, stratify = y)


def CMatrix(CM, labels=['PAY','default']):
    df=pd.DataFrame(data=CM, index = labels, columns=labels)
    df.index.name = 'TRUE'
    df.columns.name= 'Predictions'
    df.loc['Total']=df.sum()
    df['Total']=df.sum(axis=1)
    return df

metrics = pd.DataFrame(index = ['accuracy','precision','recall'],
                       columns = ['NULL','LogisticReg','ClassTree','NaiveBayes','RFC'])

#NULL model
y_pred_test = np.repeat(y_train.value_counts().idxmax(),y_test.size)
metrics.loc['accuracy','NULL'] = accuracy_score(y_pred = y_pred_test, y_true = y_test)
metrics.loc['precision','NULL'] = precision_score(y_pred = y_pred_test, y_true = y_test)
metrics.loc['recall','NULL'] = recall_score(y_pred = y_pred_test, y_true = y_test)

CM = confusion_matrix(y_pred = y_pred_test, y_true = y_test)
CMatrix(CM)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred_test = lr.predict(x_test)
metrics.loc['accuracy','LogisticReg'] = accuracy_score(y_pred = y_pred_test, y_true = y_test)
metrics.loc['precision','LogisticReg'] = precision_score(y_pred = y_pred_test, y_true = y_test)
metrics.loc['recall','LogisticReg'] = recall_score(y_pred = y_pred_test, y_true = y_test)
CM = confusion_matrix(y_pred = y_pred_test, y_true = y_test)
CMatrix(CM)

#DeciionTreeClassifer
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth = 10)
dtc.fit(x_train,y_train)
y_pred_test = dtc.predict(x_test)
metrics.loc['accuracy','ClassTree'] = accuracy_score(y_pred = y_pred_test, y_true = y_test)
metrics.loc['precision','ClassTree'] = precision_score(y_pred = y_pred_test, y_true = y_test)
metrics.loc['recall','ClassTree'] = recall_score(y_pred = y_pred_test, y_true = y_test)
CM = confusion_matrix(y_pred = y_pred_test, y_true = y_test)
CMatrix(CM)

#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(x_train,y_train)
y_pred_test = GNB.predict(x_test)
metrics.loc['accuracy','NaiveBayes'] = accuracy_score(y_pred = y_pred_test, y_true = y_test)
metrics.loc['precision','NaiveBayes'] = precision_score(y_pred = y_pred_test, y_true = y_test)
metrics.loc['recall','NaiveBayes'] = recall_score(y_pred = y_pred_test, y_true = y_test)
CM = confusion_matrix(y_pred = y_pred_test, y_true = y_test)
CMatrix(CM)

#Random Forest Classifier
from sklearn.ensemble  import RandomForestClassifier
RFC = RandomForestClassifier()
RFC.fit(x_train,y_train)
y_pred_test = RFC.predict(x_test)
metrics.loc['accuracy','RFC'] = accuracy_score(y_pred = y_pred_test, y_true = y_test)
metrics.loc['precision','RFC'] = precision_score(y_pred = y_pred_test, y_true = y_test)
metrics.loc['recall','RFC'] = recall_score(y_pred = y_pred_test, y_true = y_test)
CM = confusion_matrix(y_pred = y_pred_test, y_true = y_test)
CMatrix(CM)

fig,ax = plt.subplots(figsize = (10,5))
metrics.plot(kind='barh', ax=ax)

#Check using precision recall curve to check if naive bayes is better than LR
from sklearn.metrics import precision_recall_curve
precision_nb, recall_nb, threshold_nb = precision_recall_curve(y_true = y_test, 
                                                            probas_pred = GNB.predict_proba(x_test)[:,1])
precision_lr, recall_lr, threshold_lr = precision_recall_curve(y_true = y_test, 
                                                            probas_pred = lr.predict_proba(x_test)[:,1])

plt.plot(precision_nb,recall_nb,label='NB',color='blue')
plt.plot(precision_lr,recall_lr,label='LR',color='Red')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend()
plt.grid()

#Logistic Regression proved to be the best among all the other precition methods but
#the recall value is still lower
#therefore we should adjust the classification threshold and we can see changes

plt.plot(threshold_lr,precision_lr[1:],color='red',label='lr')
plt.plot(threshold_nb,precision_nb[1:],color='blue',label='nb')
plt.legend()
plt.grid()
plt.hlines(y=0.5,xmin=0,xmax=1,color='black')
