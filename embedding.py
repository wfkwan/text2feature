from gensim.models.keyedvectors import KeyedVectors
from config import EMBEDDING_PATH, EMBEDDING_FNAME
import os
from datetime import datetime
from collections import Counter

class WordEmbedding(object):
	def __init__(self):
		pass
		self.fpath = os.path.join(EMBEDDING_PATH, EMBEDDING_FNAME)

	def load(self, fpath=None, type_='fasttext', is_binary=None):
		if not fpath:
			fpath = self.fpath
		if not is_binary:
			is_binary = fpath.strip('/')[-4:]==".bin"

		print("Searching in %s for word vector file ..." % fpath)
		print("Using %s pretrained embedding, it may take some time ..." % type_)
		if not os.path.isfile(fpath):
		    raise IOError("Embedding file not found !")
		embeddings = KeyedVectors.load_word2vec_format(fpath, binary=is_binary)
		self.embeddings = embeddings

	def check_embedding_coverage(self, embeddings=None, list_tokens=None):
	    print("Checking Embedding coverage ...")
	    if not list_tokens:
	    	raise ValueError("Please input a list of tokens!")
	    if not embeddings:
	    	embeddings = self.embeddings
	    unmatched_token = Counter()
	    matched_token = Counter()
	    for tokens in list_tokens:
	    	for tok in tokens:
	    		if tok not in embeddings.vocab:
	    			unmatched_token[tok] += 1
	    		else:
	    			matched_token[tok] += 1
	    unmatched_token = sorted(unmatched_token.items(), key=lambda i:i[1], reverse=True)
	    with open(EMBEDDING_FNAME + datetime.strftime(datetime.now(), "_%Y%m%d_%H%M%S_unmatched_tokens"), 'w') as f:
	    	f.write("unmatched_token,count\n")
	    	for tok, count in unmatched_token:
	    		f.write("%s,%d\n" % (tok, count))
	    self.coverage = 100 * len(list(set(matched_token))) / (len(matched_token)+ len(unmatched_token))
	    print("Embedding converage is %.2f " % self.coverage, "%")
	    return matched_token, unmatched_token

	# train theembedding by ourselves instead of loading a pretrained one
	def model(self):
	    raise NotImplementedError("This part is not yet written!")

