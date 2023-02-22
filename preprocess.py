import pandas as pd
import numpy as np
import nltk
import shutil
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from json import dumps
from sklearn.feature_extraction.text import TfidfVectorizer
from libs.data_preprocessor import DataProcessor
from transformers import BertTokenizer

nltk.download('stopwords')
nltk.download('punkt')

custom_words = ['feel', 'humili', 'didnt']
keep_words = ['not']

english_words = np.array(stopwords.words('english') + custom_words)
ignore_words = np.setdiff1d(english_words, keep_words)

vectorizer = TfidfVectorizer()

dp = DataProcessor()

def process_training_file(raw, updated):
    stats = {"max": 0, "min": None}
    df = pd.read_csv(raw, header=None, names=["Sentence", "Emotion"], sep=";")

    for i in range(0, len(df)):
        string = df.at[i, 'Sentence']
        string = dp.process_line(string)
        tokens = np.setdiff1d(np.array(word_tokenize(string)), ignore_words)
        string = ' '.join(tokens)
        length = len(string)
        if not length:
            df.drop(i, inplace=True)
        else:
            stats["max"] = max(stats["max"], length)
            stats["min"] = min(stats["min"], len(
                tokens)) if stats["min"] is not None else length
            df.at[i, 'Sentence'] = string

    # Find out words have high occurence (>= 0.6)
    corpus = df.Sentence.to_numpy()
    X = vectorizer.fit_transform(corpus)
    encoded = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
    high_occurence = []
    for word in encoded:
        if encoded[word] >= 0.6:
            print(word, encoded[word])
            high_occurence.append(word)

    df.to_csv(updated, index=False, header=None, sep=";")
    return stats

shutil.copyfile("data/train_data.txt", "data/train_data_trimmed.txt")
shutil.copyfile("data/val_data.txt", "data/val_data_trimmed.txt")

stats = process_training_file("data/train_data_trimmed.txt",
                                                    "data/train_data_trimmed.txt")

print("Training", dumps(stats, indent=4))

stats = process_training_file("data/val_data_trimmed.txt",
                                                "data/val_data_trimmed.txt")

print("Validate", dumps(stats, indent=4))


