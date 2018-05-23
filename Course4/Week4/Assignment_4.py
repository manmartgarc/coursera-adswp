# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:21:12 2018

@author: manma
"""

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

path = ('X:/Google Drive/GitHub/Coursera-Applied-Data-Science-with-Python'
        '/Course4/course4_downloads/')

def convert_tag(tag):
    """
    Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets
    """

    tag_dict = {'N': 'n',
                'J': 'a',
                'R': 'r',
                'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None

# %%
def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """

    tokens = nltk.word_tokenize(doc)
    tagged = nltk.pos_tag(tokens)

    retagged = [(token, convert_tag(tag)) for token, tag in tagged]

    synsets = [wn.synsets(i,j)[0] for i,j in retagged
               if len(wn.synsets(i,j)) > 0]

    return synsets
# %%
def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity
    value.

    Sum of all of the largest similarity values and normalize this value\
    by dividing it by the number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    scores = []
    for i in s1:
        sims = []
        for j in s2:
            similarity = i.path_similarity(j)
            if similarity is not None:
                sims.append(similarity)
        if len(sims) > 0:
            scores.append(max(sims))


    return sum(scores) / len(scores)

# %%
def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) +
            similarity_score(synsets2, synsets1)) / 2

# %%
def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)
# %%
document_path_similarity('I like cats', 'I like dogs')

# %%
# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv(path + 'paraphrases.csv')
paraphrases.head()

# %%
def most_similar_docs():
    
    paraphrases['path_similarities'] = (
            paraphrases.apply(lambda row: document_path_similarity(row['D1'],
                                                                   row['D2']),
                                                                   axis=1))
    
    row = (
        paraphrases[paraphrases['path_similarities']
                    == paraphrases['path_similarities'].max()])
    
    return (row['D1'].item(),
            row['D2'].item(),
            row['path_similarities'].item())
most_similar_docs()

# %%
def label_accuracy():
    from sklearn.metrics import accuracy_score
    
    paraphrases['path_similarities'] = (
            paraphrases.apply(lambda row: document_path_similarity(row['D1'],
                                                                   row['D2']),
                                                                   axis=1))
    paraphrases['label'] = np.where(paraphrases['path_similarities'] > 0.75,
               1, 0)
    
    return accuracy_score(paraphrases['Quality'], paraphrases['label'])
label_accuracy()

# %%
import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open(path + 'newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

# %%
# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=id_map, passes=25, random_state=34)

# %%
def lda_topics():
    
    topics = ldamodel.print_topics(num_topics=10, num_words=10)
    
    return topics
lda_topics()

# %%
new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]

# %%
def topic_distribution():
    
    new_doc_trans = vect.transform(new_doc)
    new_doc_corpus = gensim.matutils.Sparse2Corpus(new_doc_trans,
                                                   documents_columns=False)
    new_doc_topics = ldamodel.get_document_topics(new_doc_corpus)
    topics_list = list(new_doc_topics)
    topics_list = topics_list[0]
    
    return topics_list
topic_distribution()

# %%
def topic_names():
    
    topic_names_doc = ['Education', 'Automobiles', 'Computers & IT',
                       'Politics', 'Travel', 'Sports', 'Health', 'Religion',
                       'Computers & IT', 'Science']
    return topic_names_doc
topic_names()
