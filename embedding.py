from gensim.models.keyedvectors import KeyedVectors
from config import EMBEDDING_PATH, EMBEDDING_FNAME
import os

class WordEmbedding(object):
	def __init__(self):
		pass
		self.fpath = os.path.join(EMBEDDING_PATH, EMBEDDING_FNAME)

	def load(self, fpath=None, type_='fasttext', is_binary=False):
		if not fpath:
			fpath = self.fpath
		print("Searching in %s for word vector file ..." % fpath)
		print("Using %s pretrained embedding, it may take some time ..." % type_)
		embeddings = KeyedVectors.load_word2vec_format(fpath, binary=is_binary)
		self.embeddings = embeddings

	# train theembedding by ourselves instead of loading a pretrained one
	def model(self):
		raise NotImplementedError("This part is not yet written!")

