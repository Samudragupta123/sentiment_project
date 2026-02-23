import string
import nltk
import pandas as pd
import torch
import numpy as np

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


# -------------------------
# Download once (safe guard)
# -------------------------
def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger_eng')


# -------------------------
# Text Cleaning Functions
# -------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def nltk_preprocess(text, mode='lemmatize'):
    if not text:
        return ""

    tokens = word_tokenize(str(text).lower())
    tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]

    if mode == 'lemmatize':
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    else:
        tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)


def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN


def process_with_pos(text):
    tokens = nltk.word_tokenize(str(text).lower())
    pos_tags = nltk.pos_tag(tokens)

    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    return " ".join(lemmatized)


# -------------------------
# MAIN BUILDER FUNCTION
# -------------------------
def build_preprocessing_objects(dataset):
    """
    Takes dataset from data_loader
    Returns fitted encoders + collate_fn
    """

    # ----- Fit TF-IDF -----
    text_corpus = dataset['text'].fillna("").astype(str)
    reason_corpus = dataset['negativereason'].fillna("").astype(str)

    all_text = pd.concat([text_corpus, reason_corpus], axis=0)

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf.fit(all_text)

    # ----- Fit OneHotEncoders -----
    sentiment_enc = OneHotEncoder(sparse_output=False).fit(dataset[['airline_sentiment']])
    airline_enc = OneHotEncoder(sparse_output=False).fit(dataset[['airline']])

    # ----- Define Collate Function (CLOSURE!) -----
    def custom_collate(batch):
        x_batch_raw, y_batch_raw = zip(*batch)

        y_encoded = torch.tensor(
            sentiment_enc.transform(np.array(y_batch_raw).reshape(-1, 1)),
            dtype=torch.float32
        )

        airlines = [[row[3]] for row in x_batch_raw]
        airline_ohe = torch.tensor(airline_enc.transform(airlines), dtype=torch.float32)

        def safe(x):
            if x is None or isinstance(x, float):
                return ""
            return str(x)

        processed_text = [
            process_with_pos(nltk_preprocess(safe(row[4])))
            for row in x_batch_raw
        ]

        processed_reason = [
            process_with_pos(nltk_preprocess(safe(row[1])))
            for row in x_batch_raw
        ]

        text_vec = torch.tensor(tfidf.transform(processed_text).toarray(), dtype=torch.float32)
        reason_vec = torch.tensor(tfidf.transform(processed_reason).toarray(), dtype=torch.float32)

        numeric_feats = torch.tensor([[row[0], row[2]] for row in x_batch_raw], dtype=torch.float32)

        return {
            "numeric_and_ohe": torch.cat([numeric_feats, airline_ohe], dim=1),
            "text_data": text_vec,
            "reason_data": reason_vec,
            "targets": y_encoded
        }

    return custom_collate, tfidf, sentiment_enc, airline_enc