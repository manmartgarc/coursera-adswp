# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 14:58:49 2018

@author: manma
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier

path = ('X:/Google Drive/GitHub/Coursera-Applied-Data-Science-with-Python/'
        'Course3/course3_downloads/')

train = pd.read_csv(path + 'train.csv',
                 encoding='ISO-8859-1')

test = pd.read_csv(path + 'test.csv',
                   encoding='ISO-8859-1')

latlons = pd.read_csv(path + 'latlons.csv')
addresses = pd.read_csv(path + 'addresses.csv')

addresses_latlons = addresses.merge(latlons, how='inner', on='address')

train = train.merge(addresses_latlons, how='inner', on='ticket_id')
test = test.merge(addresses_latlons, how='inner', on='ticket_id')

# %%
for df in [train, test]:
    X = df.select_dtypes(include=['object'])
    le = preprocessing.LabelEncoder()
    X_2 = X.astype(str).apply(le.fit_transform)

    for col in X_2.columns:
        df[col] = X_2[col]
        
    df.fillna(method='pad', inplace=True)
    df.fillna(0, inplace=True)

# %%
train_nonull = train[(train['compliance'] == 1) | (train['compliance'] == 0)]
y_train = train_nonull['compliance']

X_train = train_nonull.drop(['payment_amount', 'payment_date',
                             'payment_status', 'balance_due',
                             'collection_status', 'compliance_detail',
                             'compliance'], axis=1)
    
clf = GradientBoostingClassifier().fit(X_train, y_train)

y_score_logit = clf.decision_function(test)

probs = clf.predict_proba(test)[:,1]
test['prob'] = probs
output = test.set_index('ticket_id')['prob']

