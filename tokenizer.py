from nltk.tokenize import TreebankWordTokenizer, ToktokTokenizer
import string
from nltk.corpus import stopwords
import copy
import re
import codecs
import os

class FilteredTokenizer(object):
    # turn a doc into clean tokens
    def __init__(self):
        pass

    def punctuation_filter(self, tokens=None, punctuation=None, greedy=False):
        if not tokens:
            return tokens
        if greedy:
            punctuation = u'：•！？…。、!#$%&*+,\./:;<=>?@^\+`|~'
        if not punctuation:
            table = str.maketrans('', '', string.punctuation)
            return [w.translate(table) for w in tokens]
        else:
            regex = re.compile(punctuation)
            for tok in tokens:
                tok = re.sub(regex, tok, '')
            return tokens

    def alphabetic_filter(self, tokens=None):
        if not tokens:
            return tokens
        return [word for word in tokens if word.isalpha()]

    def stop_words_filter(self, tokens=None, stop_words=None):
        if not tokens:
            return tokens
        if not stop_words:
            stop_words = set(stopwords.words('english'))
        return [w for w in tokens if not w in stop_words]

    def customized_filter(self, tokens=None, list_excluded_text=None):
        return [w for w in tokens if not w in list_excluded_text]

    def load_custom_filter_file(self, filter_file=None):
        if not filter_file or not os.path.isfile(filter_file):
            raise IOError("Cannot find file %s !!!" %  filter_file)
        list_excluded_text = []
        with open(filter_file, 'r', encoding = 'utf8') as f:
            for line in f:
                list_excluded_text.append(line)
        return list_excluded_text

    def filter_and_tokenize(self, list_text, mode=['punctuation'], tokenizer=None, filter_fpath=None):
        if not isinstance(mode, list):
            raise ValueError("Please input a list for mode!")
        # modify this line to use other tokens
        self.tokenize(list_text, tokenizer)
        self.tokens = []
        if 'custom_filter' in mode:
            list_excluded_text = self.load_custom_filter_file(filter_fpath)

        # don't prefer stemming since the stemmed words will not be in the embedding
        for tokens in self.raw_tokens:
            for m in mode:
                if m == 'punctuation':
                    tokens = self.punctuation_filter(tokens)
                elif m == 'alphabetic':
                    tokens = self.alphabetic_filter(tokens)
                elif m == 'stop_words':
                    tokens = self.stop_words_filter(tokens)
                elif m == 'custom_filter':
                    tokens = self.customized_filter(tokens, list_excluded_text=list_excluded_text)
            self.tokens.append(tokens)
        return self.tokens

    def tokenize(self, list_text, tokenizer=None):
        if not list_text:
            return None
        if not isinstance(list_text, list):
            raise ValueError("Please input a list of string for tokenization!")
        self.list_text = list_text
        if not tokenizer:
            self.raw_tokens = [text.split() for text in list_text]
        elif "treebank" in tokenizer.lower():
            t = TreebankWordTokenizer()
            self.raw_tokens = [t.tokenize(text) for text in list_text]
        elif "toktok" in tokenizer.lower():
            t = ToktokTokenizer()
            self.raw_tokens = [t.tokenize(text) for text in list_text]

        if not self.raw_tokens:
            return None
