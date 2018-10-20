'''
Please set all the configs here
'''

EMBEDDING_PATH = '/Users/AndyKwan/Documents/data science/word_embedding/'

# facebook pretrained fasttext word embedding
EMBEDDING_FNAME = 'wiki-news-300d-1M.vec'

# google pretrained word2vec word embedding
# word2vec_fname = 'GoogleNews-vectors-negative300.bin'

DATA_PATH = "/Users/AndyKwan/Documents/COURSES/MSBD5012/project/data/"

RAW_DATA = "wine-reviews/winemag-data-130k-v2.csv"
# RAW_DATA = "wine-reviews/winemag-data_first150k.csv"

# Choices: 'punctuation', 'alphabetic', 'stop_words'
# must be a list
TOKEN_FILTERS = ['punctuation']
# None means default
TOKENIZER = None

# Whether to convert the y values into categories
CONVERT_Y = True
Y_CAT_INTERVALS = [80, 82, 84, 86, 88]
