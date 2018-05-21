# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:14:23 2018

@author: manma
"""

import pandas as pd
import numpy as np

path = ('X:/Google Drive/GitHub/Coursera-Applied-Data-Science-with-Python/'
        'Course4/course4_downloads/')

spam_data = pd.read_csv(path
                        + 'spam.csv')

spam_data['target'] = np.where(spam_data['target'] == 'spam', 1, 0)

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)

# %%
def answer_one():
    
    return spam_data['target'].mean() * 100
answer_one()

# %%
from sklearn.feature_extraction.text import CountVectorizer
def answer_two():

    vect = CountVectorizer().fit(X_train)
    longest = max(vect.get_feature_names(), key=lambda x: len(x))
    
    return longest
answer_two()

# %%
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    
    vect = CountVectorizer().fit(X_train)
    X_train_vect = vect.transform(X_train)
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train_vect, y_train)
    
    predictions = clf.predict(vect.transform(X_test))
    
    
    return roc_auc_score(y_test, predictions)
answer_three()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    
    vect = TfidfVectorizer().fit(X_train)
    feature_names = np.array(vect.get_feature_names())
    X_train_vect = vect.transform(X_train)
    
    feats_tfidf = list(zip(feature_names, X_train_vect.max(0).toarray()[0]))
    
    smallest = sorted(feats_tfidf, key=lambda x: x[1], reverse=False)[:20]
    largest = sorted(feats_tfidf, key=lambda x: x[1], reverse=True)[:20]
    
    smallest_uz = list(zip(*smallest))
    largest_uz = list(zip(*largest))
    
    s_s = pd.Series(smallest_uz[1], index=smallest_uz[0])
    l_s = pd.Series(largest_uz[1], index=largest_uz[0])
    
    return (s_s, l_s)
answer_four()

# %%
def answer_five():
    
    vect = TfidfVectorizer(min_df=4).fit(X_train)
    X_train_vect = vect.transform(X_train)
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train_vect, y_train)
    
    predictions = clf.predict(vect.transform(X_test))
    
    return roc_auc_score(y_test, predictions)
answer_five()

# %%
def answer_six():
    
    spam_mean = spam_data[spam_data['target'] == 1]['text'].str.len().mean()
    nospam_mean = spam_data[spam_data['target'] == 0]['text'].str.len().mean()
    
    return (nospam_mean, spam_mean)
answer_six()

# %%
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

from sklearn.svm import SVC

def answer_seven():
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_vect = vect.transform(X_train)
    X_train_vect = add_feature(X_train_vect, X_train.str.len())
    X_test_vect = vect.transform(X_test)
    X_test_vect = add_feature(X_test_vect, X_test.str.len())
    
    clf = SVC(C=10000)
    clf.fit(X_train_vect, y_train)
    
    predictions = clf.predict(X_test_vect)
    
    return roc_auc_score(y_test, predictions)
answer_seven()
    
# %%
def answer_eight():
    
    nospam = (
            spam_data['text'].str.findall(r'[0-9]+?').str.len()
            [spam_data['target'] == 0].mean())
    spam = (
            spam_data['text'].str.findall(r'[0-9]+?').str.len()
            [spam_data['target'] == 1].mean())
    
    return (nospam, spam)
answer_eight()

# %%
from sklearn.linear_model import LogisticRegression

def answer_nine():
    vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
    X_train_vect = vect.transform(X_train)
    X_train_vect = add_feature(X_train_vect, X_train.str.len())
    X_train_vect = (
            add_feature(X_train_vect, X_train.str.findall(r'[0-9]+?')
            .str.len()))
    
    X_test_vect = vect.transform(X_test)
    X_test_vect = add_feature(X_test_vect, X_test.str.len())
    X_test_vect = (
            add_feature(X_test_vect, X_test.str.findall(r'[0-9]+?')
            .str.len()))
    
    clf = LogisticRegression(C=100)
    clf.fit(X_train_vect, y_train)
    
    predictions = clf.predict(X_test_vect)
    
    return roc_auc_score(y_test, predictions)
answer_nine()

# %%
def answer_ten():
    
    nospam = (
            spam_data['text'].str.findall(r'\W').str.len()
            [spam_data['target'] == 0].mean())
    
    spam = (
            spam_data['text'].str.findall(r'\W').str.len()
            [spam_data['target'] == 1].mean())
    
    return (nospam, spam)
answer_ten()

# %%
def answer_eleven():
    vect = CountVectorizer(min_df=5,
                           ngram_range=(2,5),
                           analyzer='char_wb').fit(X_train)
    
    X_train_vect = vect.transform(X_train)
    X_test_vect = vect.transform(X_test)
        
    tr_len_char = X_train.str.len()
    tr_len_digs = X_train.str.findall(r'[0-9]+?').str.len()
    tr_len_nochar = X_train.str.findall(r'\W').str.len()
    
    te_len_char = X_test.str.len()
    te_len_digs = X_test.str.findall(r'[0-9]+?').str.len()
    te_len_nochar = X_test.str.findall(r'\W').str.len()
    
    X_train_vect = add_feature(X_train_vect, tr_len_char.values)
    X_train_vect = add_feature(X_train_vect, tr_len_digs.values)
    X_train_vect = add_feature(X_train_vect, tr_len_nochar.values)
    
    X_test_vect = add_feature(X_test_vect, te_len_char.values)
    X_test_vect = add_feature(X_test_vect, te_len_digs.values)
    X_test_vect = add_feature(X_test_vect, te_len_nochar.values)
    
    clf = LogisticRegression(C=100)
    clf.fit(X_train_vect, y_train)
    
    predictions = clf.predict(X_test_vect)
    
    roc = roc_auc_score(y_test, predictions)
    
    feature_names = (vect.get_feature_names()
                        + ['length_of_doc',
                           'digit_count',
                           'non_word_char_count'])
    
    feat_coef = sorted(list(zip(feature_names, clf.coef_[0])),
                       key=lambda x: x[1],
                       reverse=True)
    
    big_coef = feat_coef[:10]
    small_coef = feat_coef[-10:]
    
    big_coef = list(zip(*big_coef))
    small_coef = list(zip(*small_coef))
    
    big_coef = pd.Series(big_coef[1], index=big_coef[0])
    small_coef = pd.Series(small_coef[1], index=small_coef[0])
           
    return (roc, small_coef.sort_values(), big_coef)
answer_eleven()
    
    