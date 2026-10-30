"""
Shared text-cleaning pipeline.

IMPORTANT: this exact sequence of steps is used both when training the
model (train_model.py / train_model.ipynb) and when the Streamlit app
(app.py) cleans a new piece of text before asking the model to predict
on it. If the two ever go out of sync, predictions will be unreliable,
so keep all preprocessing logic here in one place and import it from
both scripts.

This version has no NLTK dependency (no downloads, no data-path issues,
no version incompatibilities) -- it uses a built-in English stopword
list and a simple whitespace/punctuation-aware tokenizer instead.
"""

import re
import string

# A standard English stopword list (same words NLTK's 'english' stopword
# corpus ships with), hardcoded so nothing needs to be downloaded.
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "that'll", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "don't", "should", "should've", "now",
    "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't",
    "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn",
    "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't",
    "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't",
    "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't",
    "weren", "weren't", "won", "won't", "wouldn", "wouldn't",
}


def ensure_nltk_data():
    """Kept for backwards compatibility with earlier app.py / notebook
    versions that call this on startup. There's nothing to download
    anymore, so this is a no-op."""
    pass


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    return "".join(ch for ch in text if not ch.isdigit())


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


def remove_emoji(text: str) -> str:
    return "".join(ch for ch in text if ch.isascii())


def remove_stopwords(text: str) -> str:
    words = text.split()
    return " ".join(w for w in words if w not in STOP_WORDS)


def clean_text(text: str) -> str:
    """Run the full cleaning pipeline on a single string, in the same
    order used during training."""
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    return text