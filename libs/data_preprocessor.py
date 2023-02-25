'''
    Filename: data_preprocessor.py
    Usage: Provide the stemming and lemmatization functions.

    Example:
        # Instance method
        import pandas as pd
        from libs.data_preprocessor import DataProcessor
        
        dp = DataProcessor()
        df = pd.DataFrame(
            [{
                "Sentence": "Today is a beautful day, isn't it?", 
                "Emotion": "joy"
            }], 
            columns=["Sentence", "Emotion"]
        )
        processed = dp.process(df, ["Sentence", "Emotion"])
        print(processed)

        # Output
                                Sentence Emotion
        0  today is a beaut day isn t it     joy
'''

import nltk
import pandas as pd
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

nltk.download("wordnet")

class DataProcessor:
    # Initialized with the tokenizer, stemmer and lemmatizer    
    def __init__(self):
        self.stemmer = EnglishStemmer()
        self.lemma = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r"\w+")

    # Perform tokenization, lemmatization, and stemming
    def process_words(self, words):
        buff = []
        for word in self.tokenizer.tokenize(words):            
            word = self.lemma.lemmatize(word)
            word = self.stemmer.stem(word)
            buff.append(word)
        return " ".join(buff)
    
    # Loop through the DataFrame and process the words
    def process(self, lines, columns):        
        buff = []
        for [words, feat] in lines.to_numpy().tolist():
            words = self.process_words(words)
            buff.append([words, feat])
        return pd.DataFrame(buff, columns=columns)
    
    # Lazy method for processing a line
    @classmethod
    def process_line(cls, line):
        return cls().process_words(line)
