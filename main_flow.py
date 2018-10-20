'''
This is a demo of how to use the code
'''

from tokenizer import FilteredTokenizer
from embedding import WordEmbedding
from transformer import Transformer
from config import (EMBEDDING_PATH, EMBEDDING_FNAME, DATA_PATH, RAW_DATA, 
					TOKEN_FILTERS, TOKENIZER, Y_CAT_INTERVALS, CONVERT_Y)
import os
import pandas as pd

fpath = os.path.join(DATA_PATH, RAW_DATA)
df = pd.read_csv(fpath)
descriptions = df['description'].tolist()

FT = FilteredTokenizer()
Tokens = FT.filter_and_tokenize(descriptions, mode=TOKEN_FILTERS, tokenizer=TOKENIZER)

WordEmbedding_ = WordEmbedding()
WordEmbedding_.load()

# Examples of things you can do with the embeddings
print(WordEmbedding_.word_vectors.most_similar(positive=['woman', 'king'], negative=['man']))
print(WordEmbedding_.word_vectors.most_similar("dont"))
print(WordEmbedding_.word_vectors.most_similar("a"))

matched_tokens, unmatched_tokens = WordEmbedding_.check_embedding_coverage(list_tokens=Tokens)
# Then you will get a file named <embedding file name> + <date time> + unmatched tokens
# this is a file with count distinct unmatched tokens, sorted in descending order

# Then you are able to see these attributes:
print("WordEmbedding_.coverage", WordEmbedding_.coverage)
print("WordEmbedding_.wordvec_map", WordEmbedding_.wordvec_map)
print("You can get a word vector of the word 'hello' by calling: WordEmbedding_.wordvec_map['hello']", 
	  WordEmbedding_.wordvec_map['hello'])

from transformer import Transformer
T = Transformer(fasttext_word_embedding.wordvec_map)

y = df['points'].tolist()
X_new, y_new = T.fit_transform(Tokens, y, drop_long_sentences=70,
                    drop_short_sentences=5, num2cat_=CONVERT_Y, intervals=Y_CAT_INTERVALS)