# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:23:24 2018

@author: manma
"""

import nltk
import pandas as pd
import numpy as np

path = ('C:/Users/manma/Google Drive/GitHub/'
        'Coursera-Applied-Data-Science-with-Python/Course4/course4_downloads/')

with open(path + 'moby.txt', 'r') as f:
    moby_raw = f.read()
    
moby_tokens = nltk.word_tokenize(moby_raw)
    
# %%
def example_one():
    
    return len(nltk.word_tokenize(moby_raw))

example_one()

# %%
def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) 

example_two()

# %%
from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in moby_tokens]

    return len(set(lemmatized))

example_three()

# %%
def answer_one():
    
    unique_count = example_two()
    total_count = example_one()
    
    return unique_count / total_count

answer_one()

# %%
def answer_two():
    
    dist = nltk.FreqDist(moby_tokens)
    
    whales = dist['whale'] + dist['Whale']
    total = sum(dist.values())
    
    return (whales / total) * 100
answer_two()
# %%
def answer_three():
    #lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(moby_raw)
    dist = nltk.FreqDist(tokens)       
    
    return dist.most_common(20)

answer_three()

# %%
def answer_four():
    
    tokens = nltk.word_tokenize(moby_raw)
    dist = nltk.FreqDist(tokens)
    
    selected = [w for w in dist.keys() if len(w) > 5 and dist[w] > 150]
    
    return sorted(selected)

answer_four()

# %%
def answer_five():
    
    lengths = [(w, len(w)) for w in moby_tokens]
    
    return max(lengths, key=lambda x: x[1])

answer_five()

# %%
def answer_six():
    
    dist = nltk.FreqDist(moby_tokens)
    frequent = [(f, w) for w,f in dist.items() if w.isalpha() == True and f > 2000]
    
    return sorted(frequent, key=lambda x: x[0], reverse=True)

answer_six()

# %%
def answer_seven():
    
    sents = nltk.sent_tokenize(moby_raw)
    sent_toks = [len(nltk.word_tokenize(sent)) for sent in sents]
    
    return sum(sent_toks) / len(sent_toks)

answer_seven()

# %%
def answer_eight():
    
    pos = nltk.pos_tag(moby_tokens)
    
    only_pos = [item[1] for item in pos]
    
    dist_pos = nltk.FreqDist(only_pos)
    
    return dist_pos.most_common(5)

answer_eight()

# %%
from nltk.corpus import words

correct_spellings = words.words()

# %%
def jaccard_d(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    dist = len(set1 ^ set2) / max(1, len(set1 | set2))
    
    return dist

# %%
def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
           
    recoms = []
    
    for entry in entries:
        possibles = [w for w in correct_spellings if w[0] == entry[0]]
        recoms.append(min(possibles,
                          key=lambda x: jaccard_d(nltk.ngrams(entry, n=3),
                                                  nltk.ngrams(x, n=3))))
    
    return recoms
answer_nine()

# %%
def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
       
    recoms = []
    
    for entry in entries:
        possibles = [w for w in correct_spellings if w[0] == entry[0]]
        recoms.append(min(possibles,
                          key=lambda x: jaccard_d(nltk.ngrams(entry, n=4),
                                                  nltk.ngrams(x, n=4))))
    
    return recoms
answer_ten()

# %%
def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    recoms = []
    
    for entry in entries:
        possibles = [w for w in correct_spellings if w[0] == entry[0]]
        recoms.append(min(possibles,
                          key=lambda x: (
                                  nltk.edit_distance(entry,
                                                     x,
                                                     transpositions=True))))
        
    return recoms
answer_eleven()
        
        
