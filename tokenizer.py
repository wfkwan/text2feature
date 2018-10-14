from nltk.tokenize import TreebankWordTokenizer, ToktokTokenizer
import string
from nltk.corpus import stopwords
import re
import copy
 
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
        else:
            return [w for w in tokens if not w in stop_words]
        
    def short_tokens_filter(self, tokens=None, length=1):
        if not tokens:
            return tokens
        return [word for word in tokens if len(word) > length]
    
    def long_tokens_filter(self, tokens=None, length=70):
        if not tokens:
            return tokens
        return [word for word in tokens if len(word) < length]
    
    def filter_and_tokenize(self, list_text):
        # modify this line to use other tokens
        self.tokenize(list_text)
        self.tokens = []
        for tok in self.raw_tokens:
            tokens = self.punctuation_filter(tok)
    #         tokens = self.alphabetic_filter(self.tokens)
    #         tokens = self.stop_words_filter(self.tokens)
    #         tokens = self.short_tokens_filter(self.tokens)
    #         tokens = self.long_tokens_filter(self.tokens)
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
