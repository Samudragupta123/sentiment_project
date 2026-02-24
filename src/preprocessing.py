import string
import nltk
import pandas as pd
import torch
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.preprocessing import OneHotEncoder

# -------------------------
# NLTK Setup
# -------------------------
def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt_tab')

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
# Vocabulary and Encoding
# -------------------------
def build_vocab(corpus, min_freq=1):
    tokens = [word for text in corpus for word in text.split()]
    counter = Counter(tokens)
    vocab = {word: i+1 for i, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab["<UNK>"] = len(vocab) + 1
    return vocab

def encode_text(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]

# -------------------------
# Main Preprocessing Builder
# -------------------------
def build_preprocessing_objects(dataset, max_seq_len=50, min_freq=1):
    """
    Returns:
        collate_fn: function to use in DataLoader
        vocab: word2idx dictionary
        sentiment_enc, airline_enc: fitted OneHotEncoders
    """

    # ----------------- Process Text Corpus -----------------
    text_corpus = dataset['text'].fillna("").astype(str).apply(lambda x: process_with_pos(nltk_preprocess(x)))
    reason_corpus = dataset['negativereason'].fillna("").astype(str).apply(lambda x: process_with_pos(nltk_preprocess(x)))

    all_text = pd.concat([text_corpus, reason_corpus], axis=0)
    
    # Build vocabulary
    vocab = build_vocab(all_text, min_freq=min_freq)

    # ----------------- One-Hot Encoders -----------------
    sentiment_enc = OneHotEncoder(sparse_output=False).fit(dataset[["airline_sentiment"]])
    airline_enc = OneHotEncoder(sparse_output=False).fit(dataset[["airline"]])

    # ----------------- Collate Function -----------------
    def collate_fn(batch):
        x_batch_raw, y_batch_raw = zip(*batch)

        # Targets
        y_df = pd.DataFrame(y_batch_raw, columns=["airline_sentiment"])
        y_encoded = torch.tensor(sentiment_enc.transform(y_df), dtype=torch.float32)

        # Airline OHE
        airlines = pd.DataFrame([row[3] for row in x_batch_raw], columns=["airline"])
        airline_ohe = torch.tensor(airline_enc.transform(airlines), dtype=torch.float32)

        # Numeric features
        numeric_feats = torch.tensor([[row[0], row[2]] for row in x_batch_raw], dtype=torch.float32)

        # Safe text
        def safe(x):
            if x is None or isinstance(x, float):
                return ""
            return str(x)

        # Encode sequences
        processed_text = [process_with_pos(nltk_preprocess(safe(row[4]))) for row in x_batch_raw]
        processed_reason = [process_with_pos(nltk_preprocess(safe(row[1]))) for row in x_batch_raw]

        text_seq = [torch.tensor(encode_text(text, vocab))[:max_seq_len] for text in processed_text]
        text_seq = pad_sequence(text_seq, batch_first=True, padding_value=0)

        reason_seq = [torch.tensor(encode_text(text, vocab))[:max_seq_len] for text in processed_reason]
        reason_seq = pad_sequence(reason_seq, batch_first=True, padding_value=0)

        return {
            "numeric_and_ohe": torch.cat([numeric_feats, airline_ohe], dim=1),
            "text_seq": text_seq,
            "reason_seq": reason_seq,
            "targets": y_encoded
        }

    return collate_fn, vocab, sentiment_enc, airline_enc