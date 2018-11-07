from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
import re
import os


class Transformer(object):
	
	def __init__(self, word2vec=None, use_count_vectorizer=False):
		if not word2vec:
			raise ValueError("Please input a word to vector mapping!")
		if not use_count_vectorizer:
			self.vectorizer = TfidfEmbeddingVectorizer(word2vec)
		else:
			self.vectorizer = MeanEmbeddingVectorizer(word2vec)

	def fit_transform(self, X, y, drop_short_sentences=None, drop_long_sentences=None, 
					  num2cat_=False, intervals=None):
		self.fit(X, y, drop_short_sentences, drop_long_sentences)
		return self.transform(self.X, self.y, num2cat_, intervals)

	def fit(self, X, y, drop_short_sentences=None, drop_long_sentences=None):
		self.vectorizer.fit(X, y, drop_short_sentences, drop_long_sentences)
		self.row2doc = self.vectorizer.row2doc
		self.X = self.vectorizer.X
		self.y = self.vectorizer.y
		return self

	def transform(self, X, y, num2cat_=False, intervals=None):
		self.intervals = intervals
		return self.vectorizer.transform(X, y, num2cat_, intervals)
			

# Tfidf-based vectorizer
'''
	input: a list of sentences with a list of tokens in each sentence, 
	will automatically deal with 
	-  sentences without tokens and
	-  too long or too short sentences based on number of tokens' restriction in the fit function

'''
class TfidfEmbeddingVectorizer(object):
    
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec)

    def fit(self, X, y, drop_short_sentences=None, drop_long_sentences=None, 
    	    num2cat_=False, intervals=None):
    	# row number of the matrix to mapping of document number
    	row2doc = {}
    	row = 0
    	X_filtered = []
    	y_filtered = []
    	total = len(X)
    	for ind, doc, cat in zip(range(total), X, y):
    		if doc:
    			if drop_short_sentences and len(doc) < drop_short_sentences:
    				continue
    			if drop_long_sentences and len(doc) > drop_long_sentences:
    				continue
    			X_filtered.append(doc)
    			y_filtered.append(cat)
    			row2doc[row] = ind
    		row += 1
    	self.row2doc = row2doc
    	tfidf = TfidfVectorizer(analyzer=lambda x: x)
    	tfidf.fit(X_filtered)
    	# if a word was never seen - it must be at least as infrequent
    	# as any of the known words - so the default idf is the max of 
    	# known idf's
    	max_idf = max(tfidf.idf_)
    	self.word2weight = defaultdict(lambda: max_idf,
    								   [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    	self.X = X_filtered
    	self.y = y_filtered
    	return self

    def transform(self, X, y, num2cat_=False, intervals=None):
    	if num2cat_ and not intervals:
    		raise ValueError("Please input intervals to transform y!")
    	if num2cat_:
    		y = num2cat(y, intervals)
    		y = to_categorical(y, len(set(y))).astype(int)
    	return np.array([np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                		for words in X
            			]), y

# count based vectorizer
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim=None):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        if not dim:
            print("Automatically set dimension of vectors to be 300.")
            self.dim = 300
        else:
            dim = len(set([v.shape[0] for k,v in word2vec.items()]))
            if  dim > 1:
                raise ValueError("Nonunique dimensions found on the vectors")
            else:
                self.dim = dim

    def fit(self, X, y, drop_short_sentences=None, drop_long_sentences=None):
    	# row number of the matrix to mapping of document number
    	row2doc = {}
    	row = 0
    	X_filtered = []
    	y_filtered = []
    	total = len(X)
    	for ind, doc, cat in zip(range(total), X, y):
    		if doc:
    			if drop_short_sentences and len(doc) < drop_short_sentences:
    				continue
    			if drop_long_sentences and len(doc) > drop_long_sentences:
    				continue
    			X_filtered.append(doc)
    			y_filtered.append(cat)
    			row2doc[row] = ind
    		row += 1
    	self.row2doc = row2doc
    	self.X = X_filtered
    	self.y = y_filtered
    	return self

    def transform(self, X, y, num2cat_=False, intervals=None):
    	if num2cat_ and not intervals:
    		raise ValueError("Please input intervals to transform y!")
    	if num2cat_:
    		y = num2cat(y, intervals)
    	y = to_categorical(y, len(set(y))).astype(int)
    	return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    	or [np.zeros(self.dim)], axis=0)
            			for words in X
        				]), y

def num2cat(list_of_numbers=None, list_of_intervals=None):
	if not list_of_intervals or not list_of_numbers:
		raise ValueError("Please input a list of numbers and the corresponding intervals " 
						 "for numbers to categories convertion!")
	#return pd.cut(list_of_numbers, list_of_intervals).labels.tolist()
	return pd.cut(list_of_numbers, list_of_intervals).codes.tolist()

def write_csv(X, y):
	if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
		ind = 0
		for file in [i for i in os.walk('.')][0][2]:
		    if re.match("X%d.csv" % ind, file):
		        ind += 1
		if ind != 0:
		    np.savetxt("X%d.csv" % ind, X, delimiter=",")
		    np.savetxt("y%d.csv" % ind, y, delimiter=",")
		else:
		    np.savetxt("X.csv", X, delimiter=",")
		    np.savetxt("y.csv", y, delimiter=",")
	else:
		raise ValueError("Make sure that you have inputed numpy arrays!")




