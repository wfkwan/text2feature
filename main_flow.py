'''
This is a demo of how to use the code
'''

from tokenizer import FilteredTokenizer
from embedding import WordEmbedding
from transformer import Transformer
from config import (EMBEDDING_PATH, EMBEDDING_FNAME, DATA_PATH, RAW_DATA, 
					TOKEN_FILTERS, TOKENIZER, Y_CAT_INTERVALS, CONVERT_Y, DROP_SHORT_SENTENCES,
					DROP_LONG_SENTENCES)
import os
import pandas as pd
from transformer import Transformer

fpath = os.path.join(DATA_PATH, RAW_DATA)
df = pd.read_csv(fpath)
descriptions = df['description'].tolist()

FT = FilteredTokenizer()
Tokens = FT.filter_and_tokenize(descriptions, mode=TOKEN_FILTERS, tokenizer=TOKENIZER)

WordEmbedding_ = WordEmbedding()
WordEmbedding_.load()

print("====== Examples of things you can do with the embeddings =======")
print(WordEmbedding_.word_vectors.most_similar(positive=['woman', 'king'], negative=['man']))
print(WordEmbedding_.word_vectors.most_similar("dont"))
print(WordEmbedding_.word_vectors.most_similar("a"))

matched_tokens, unmatched_tokens = WordEmbedding_.check_embedding_coverage(list_tokens=Tokens, verbose=True)
# Then you will get a file named <embedding file name> + <date time> + unmatched tokens
# this is a file with count distinct unmatched tokens, sorted in descending order

# Then you are able to see these attributes:
print("WordEmbedding_.coverage", WordEmbedding_.coverage)
# print("WordEmbedding_.wordvec_map", WordEmbedding_.wordvec_map)
print("You can get a word vector of the word 'hello' by calling: WordEmbedding_.word_vectors.get_vector('hello')", 
	  WordEmbedding_.word_vectors.get_vector('hello'))

T = Transformer(WordEmbedding_.wordvec_map)

# will convert the points numbers (score values) into one-hot vectors of categories defined by us (interval)
# You can change the setting in config
y = df['points'].tolist()
X, y = T.fit_transform(Tokens, y, drop_long_sentences=DROP_LONG_SENTENCES,
                    drop_short_sentences=DROP_SHORT_SENTENCES, num2cat_=CONVERT_Y, intervals=Y_CAT_INTERVALS)
print("X.shape, y.shape ", X.shape, y.shape)