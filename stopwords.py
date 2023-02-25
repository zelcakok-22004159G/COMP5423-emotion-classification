from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import nltk
import shutil
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from json import dumps

from libs.data_preprocessor import DataProcessor

ignore_words = np.array(stopwords.words('english'))

dp = DataProcessor()

vectorizer = TfidfVectorizer(
    stop_words=ignore_words.tolist(),
    preprocessor=lambda text: text,
    token_pattern='\w+',
    binary=True,
    max_df=1e-4
)  

df = pd.read_csv("data/train_data.txt", header=None, names=["Sentence", "Emotion"], sep=";")
df.Sentence.apply(dp.process_line)

X = vectorizer.fit_transform(df.Sentence.tolist())
r = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

pd.DataFrame(r.columns.values).to_csv("data/stopwords.txt", header=None, index=False)
