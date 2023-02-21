import nltk
import pandas as pd
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# nltk.download("wordnet")

class DataProcessor:
    def __init__(self):
        self.stemmer = EnglishStemmer()
        self.lemma = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r"\w+")

    def process_words(self, words):
        buff = []
        for word in self.tokenizer.tokenize(words):            
            word = self.lemma.lemmatize(word)
            word = self.stemmer.stem(word)
            buff.append(word)
        return " ".join(buff)
    
    def process(self, lines, columns):        
        buff = []
        for [words, feat] in lines.to_numpy().tolist():
            words = self.process_words(words)
            buff.append([words, feat])
        return pd.DataFrame(buff, columns=columns)