from gensim.models.keyedvectors import KeyedVectors
from config import EMBEDDING_PATH, EMBEDDING_FNAME
import os
from datetime import datetime
from collections import Counter
from transformer import Transformer

class WordEmbedding(KeyedVectors):
	def __init__(self):
		pass

	def load(self, fpath=None, is_binary=False):
		fpath = os.path.join(EMBEDDING_PATH, EMBEDDING_FNAME)
		if not os.path.isfile(fpath):
			raise IOError("Embedding file not found, please check!")
		self.fpath = fpath

		if not is_binary:
			is_binary = fpath.strip('/')[-4:]==".bin"

		print("Searching in %s for word vector file ..." % fpath)
		print("Loading pretrained embedding, it may take some time ...")
		embeddings = KeyedVectors.load_word2vec_format(fpath, binary=is_binary)
		self.word_vectors = embeddings
		self.vocab = embeddings.vocab
		self.vectors = embeddings.vectors

	def check_embedding_coverage(self, embeddings=None, list_tokens=None, verbose=False):
	    print("Checking Embedding coverage ...")
	    if not list_tokens:
	    	raise ValueError("Please input a list of tokens!")
	    if not embeddings:
	    	embeddings = self.word_vectors
	    unmatched_token = Counter()
	    matched_token = Counter()
	    wordvec_map = {}
	    for tokens in list_tokens:
	    	for tok in tokens:
	    		if tok not in embeddings.vocab:
	    			unmatched_token[tok] += 1
	    		else:
	    			matched_token[tok] += 1
	    			wordvec_map[tok] = embeddings[tok]
	    self.wordvec_map = wordvec_map

	    unmatched_token = sorted(unmatched_token.items(), key=lambda i:i[1], reverse=True)
	    with open(EMBEDDING_FNAME + datetime.strftime(datetime.now(), "_%Y%m%d_%H%M%S_unmatched_tokens"), 'w') as f:
	    	f.write("unmatched_token,count\n")
	    	for tok, count in unmatched_token:
	    		f.write("%s,%d\n" % (tok, count))
	    self.uniq_dist_coverage = 100 * len(list(set(matched_token))) / (len(matched_token) + len(unmatched_token))

	    #### TODO: render it
	    num_matched_tokens = sum([v for k,v in matched_token.items()])
	    num_unmatched_tokens = sum([v for k,v in unmatched_token])
	    self.coverage = 100 * num_matched_tokens / (num_matched_tokens + num_unmatched_tokens)
	    if verbose:
	    	print("Embedding converage (in terms of unique distinct tokens) is %.2f " % self.uniq_dist_coverage, "%")
	    	print("Embedding converage (in terms of number of occurance of tokens) is %.2f " % self.coverage, "%")
	    return matched_token, unmatched_token

	# train theembedding by ourselves instead of loading a pretrained one
	def trainable_embedding_layer(self):
	    raise NotImplementedError("This part is not yet written!")
	    # self.get_keras_embedding()

