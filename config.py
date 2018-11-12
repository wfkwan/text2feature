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

# Choices: 'punctuation', 'alphabetic', 'stop_words', 'custom_filter'
# must be a list
TOKEN_FILTERS = ['punctuation', 'custom_filter']
# used only when custom_filter is specified for TOKEN_FILTERS
CUSTOM_FILTER_PATH = "/Users/AndyKwan/Documents/COURSES/MSBD5012/project/code/filter.txt"
# None means default
TOKENIZER = None
DROP_SHORT_SENTENCES = 5
DROP_LONG_SENTENCES = 70

# Whether to convert the y values into categories
CONVERT_Y = True
Y_CAT_INTERVALS = [79.5, 83.5, 87.5, 91.5, 95.5]

