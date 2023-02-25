'''
    Filename: stopwords.py
    Usage: Find out the stopwords using TF-IDF
'''
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

from libs.data_preprocessor import DataProcessor

# Load the existing known stopwords
ignore_words = np.array(stopwords.words('english'))

dp = DataProcessor()

'''
    Create the vector creator and only 
    contains the words that the tf-idf score <= 1e-4
'''
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

# Save to data/stopwords.txt
pd.DataFrame(r.columns.values).to_csv("data/stopwords.txt", header=None, index=False)
